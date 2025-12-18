# run_experiment.py

"""
Main entry point for PEARL + C-ALM experiments.
Supports:
1. Vanilla ALM (Baseline)
2. PEARL Meta-Training (GPU-accelerated)
3. PEARL Evaluation (Inference on specific instances)
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
import torch  # 添加 torch 以检测 GPU

# --- Imports ---
from src.adaptive_quantum_cvrp.common.cvrp_instance import CVRPInstance
from src.adaptive_quantum_cvrp.alm.optimizer import ALMOptimizer
from src.adaptive_quantum_cvrp.alm.classical_solver import ClassicalSolver

# --- 新增的 PEARL 相关引用 (假设文件路径如下) ---
from src.adaptive_quantum_cvrp.rl.pearl_agent import PEARLAgent
from src.adaptive_quantum_cvrp.rl.meta_env import MetaCVRPEnv
# 你需要一个 Replay Buffer，假如你还没有，可以暂时用简单的 list 或自己写的 buffer
from src.adaptive_quantum_cvrp.rl.replay_buffer import MetaReplayBuffer

from src.adaptive_quantum_cvrp.utils.config_loader import load_config
from src.adaptive_quantum_cvrp.utils.logging_config import setup_logging

# 检测 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_vanilla_alm(config, instance, solver, output_dir):
    """(保持不变) Runs a standard ALM optimization without RL."""
    logging.info(f"Starting Vanilla ALM experiment for instance: {instance.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    alm_config = config["alm"]
    optimizer = ALMOptimizer(instance, alm_config["max_iterations"], solver)

    results = optimizer.solve(alm_config["initial_penalty_mu"])

    logging.info(f"Best feasible cost for {instance.name}: {results['best_cost']}")

    solution_path = output_dir / "solution.json"
    with open(solution_path, "w") as f:
        json.dump({
            "instance": instance.name,
            "cost": results['best_cost'],
            "routes": results['best_solution'].routes if results['best_solution'] else "None"
        }, f, indent=2)


def run_pearl_meta_training(config, solver, base_output_dir):
    """
    【核心修改】PEARL 元训练循环 (运行在 GPU 上)
    这里不针对具体文件，而是使用生成器生成多任务数据。
    """
    logging.info(f"Starting PEARL Meta-Training on device: {DEVICE}")

    pearl_config = config["pearl"]
    meta_env = MetaCVRPEnv(solver=solver)  # 环境需要 solver 来跑 ALM

    # 初始化 Agent，传入 GPU device
    agent = PEARLAgent(
        state_dim=meta_env.observation_space.shape[0],
        action_dim=meta_env.action_space.shape[0],
        latent_dim=pearl_config["latent_dim"],
        hidden_dim=pearl_config["hidden_dim"],
        lr=pearl_config["learning_rate"],
        device=DEVICE  # <--- 关键：传入 cuda
    )

    replay_buffer = MetaReplayBuffer(
        max_size=pearl_config["buffer_size"],
        state_dim=meta_env.observation_space.shape[0],
        action_dim=meta_env.action_space.shape[0],
        latent_dim=pearl_config["latent_dim"],
        device=DEVICE
    )

    num_iterations = pearl_config["num_iterations"]
    meta_batch_size = pearl_config["meta_batch_size"]
    num_context_episodes = pearl_config["num_context_episodes"]
    num_train_episodes = pearl_config["num_train_episodes"]

    # --- Meta-Training Loop ---
    for iteration in range(num_iterations):
        # 1. 采样任务 (Sample Tasks)
        # 假设 env.tasks 返回任务总数
        task_indices = np.random.choice(len(meta_env.tasks), meta_batch_size)

        # 数据收集 (Data Collection)
        for idx in task_indices:
            meta_env.reset_task(idx)
            agent.clear_context()  # 清除上一个任务的 Context

            # 1.1 积累 Context (Posterior Sampling)
            # 在这个阶段，Agent 尝试理解当前是哪种 CVRP 分布
            for _ in range(num_context_episodes):
                obs = meta_env.reset()  # 生成一个新的随机实例
                agent.infer_z()  # 基于当前 context 推断 z

                # 采样动作 (Exploration)
                action = agent.select_action(obs, deterministic=False)

                # 执行 C-ALM 求解 (这一步在 CPU 上跑)
                next_obs, reward, done, _ = meta_env.step(action)

                # 更新 Context
                agent.update_context(obs, action, reward)

            # 1.2 收集训练数据 (Traj Collection)
            # 现在 Agent 已经根据 Context "理解" 了任务，开始正式跑数据存入 Buffer
            agent.infer_z()  # 再次推断 z (此时 Context 更丰富了)

            for _ in range(num_train_episodes):
                obs = meta_env.reset()
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, done, _ = meta_env.step(action)

                # 存入 Buffer (带上 task_id 以便后续提取 context)
                replay_buffer.add(task_idx=idx,
                                  state=obs, action=action, reward=reward,
                                  next_state=next_obs, done=done)

        # 2. 模型更新 (Meta-Optimization on GPU)
        loss_info = {}
        for _ in range(pearl_config["num_gradient_steps"]):
            # 从 Buffer 中采样 Batch
            # context_batch: 用于 Encoder
            # transition_batch: 用于 Actor/Critic
            context_batch, transition_batch = replay_buffer.sample_batch(meta_batch_size)

            loss_dict = agent.train_step(context_batch, transition_batch)
            loss_info.update(loss_dict)

        if iteration % 10 == 0:
            logging.info(f"Iter {iteration}/{num_iterations} | "
                         f"Actor Loss: {loss_info.get('actor_loss', 0):.4f} | "
                         f"Critic Loss: {loss_info.get('critic_loss', 0):.4f} | "
                         f"KL: {loss_info.get('kl', 0):.4f}")

        # 定期保存模型
        if iteration % 50 == 0:
            save_path = base_output_dir / "checkpoints"
            save_path.mkdir(exist_ok=True)
            agent.save(save_path / f"pearl_agent_{iteration}.pth")

    logging.info("PEARL Meta-Training Complete.")
    return agent  # 返回训练好的 Agent


def run_pearl_evaluation(agent, config, instance_paths, solver, base_output_dir):
    """
    【评估阶段】使用训练好的 PEARL Agent 在具体 Benchmark 文件上测试
    """
    logging.info("Starting PEARL Evaluation on Benchmark Instances...")

    # 确保 Agent 在评估模式
    agent.clear_context()

    # 创建一个单纯用来跑具体实例的环境 wrapper
    # 这里我们不需要 MetaEnv 的生成功能，只需要它的 step 逻辑
    from src.adaptive_quantum_cvrp.rl.environment import ALMPenaltyEnv

    for instance_path in instance_paths:
        instance_name = instance_path.stem
        instance_output_dir = base_output_dir / instance_name
        instance_output_dir.mkdir(parents=True, exist_ok=True)

        instance = CVRPInstance.from_file(instance_path)

        # 初始化单实例环境
        # max_alm_steps 需要和训练时保持一致或更大
        env = ALMPenaltyEnv(instance, solver, max_steps=config["rl"]["max_alm_steps"])

        # --- PEARL 推理流程 ---
        # 1. Context Adaptation (适应阶段)
        # 即使是测试，PEARL 也需要先"试错"几次来推断 z
        # 对于固定实例，我们可以让它跑几次 ALM 作为 Context
        agent.clear_context()

        logging.info(f"Adapting to {instance_name}...")
        adaptation_steps = config["pearl"].get("num_inference_steps", 5)

        # 临时 Context 收集循环
        for _ in range(adaptation_steps):
            obs, _ = env.reset()
            agent.infer_z()
            # 这里可以用 deterministic=False 来增加探索性，探测该实例特性
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update_context(obs, action, reward)

        # 2. Final Execution (正式执行阶段)
        logging.info(f"Solving {instance_name} with adapted parameters...")
        agent.infer_z()  # 生成最终的 Latent Context

        obs, _ = env.reset()
        # 这次使用确定性策略，给出最优参数
        action = agent.select_action(obs, deterministic=True)

        # 执行完整的 ALM
        # 注意：这里我们只看一步 step，因为你的 ALM 环境把整个优化过程封装在一个 step 里了
        # 如果你的 env.step 是迭代式的，这里需要写 while loop
        next_obs, reward, terminated, truncated, info = env.step(action)

        best_cost = info.get('cost', float('inf'))
        best_routes = info.get('solution', None)

        logging.info(f"Instance: {instance_name} | Action(rho, sigma): {action} | Cost: {best_cost}")

        # 保存结果
        solution_path = instance_output_dir / "solution_pearl.json"
        with open(solution_path, "w") as f:
            json.dump({
                "instance": instance.name,
                "cost": best_cost,
                "parameters": [float(x) for x in action],
                "routes": best_routes.routes if best_routes else "None"
            }, f, indent=2)


def run(config_path: Path):
    """
    Main Dispatcher
    """
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    base_output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(base_output_dir, config["experiment"]["log_level"])
    logging.info(f"Loaded configuration. GPU Available: {torch.cuda.is_available()}")

    # 1. Setup Solver (FORCE CLASSICAL)
    # 强制使用经典求解器，忽略配置文件里的 quantum 选项
    solver = ClassicalSolver()
    logging.info("Forced Classical Subproblem Solver for PEARL architecture.")

    # 2. Identify Experiment Type
    exp_type = config["experiment"]["type"]

    if exp_type == "vanilla_alm":
        # Load instances (Old logic)
        instance_paths = get_instance_paths(config)
        for path in instance_paths:
            instance = CVRPInstance.from_file(path)
            run_vanilla_alm(config, instance, solver, base_output_dir / path.stem)

    elif exp_type == "pearl_train":
        # Phase 1: Meta-Training (Generative, GPU intensive)
        trained_agent = run_pearl_meta_training(config, solver, base_output_dir)

        # Phase 2: Evaluation on Benchmarks (Optional but recommended)
        if "instance_folder" in config["data"]:
            instance_paths = get_instance_paths(config)
            run_pearl_evaluation(trained_agent, config, instance_paths, solver, base_output_dir)

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    logging.info("--- All processing complete. ---")


def get_instance_paths(config):
    """Helper to get instance list"""
    instance_paths = []
    if "instance_folder" in config["data"]:
        folder = Path(config["data"]["instance_folder"])
        instance_paths = sorted(list(folder.glob("*.vrp")))
    elif "instance_path" in config["data"]:
        instance_paths.append(Path(config["data"]["instance_path"]))
    return instance_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PEARL-ALM experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file.")
    args = parser.parse_args()
    run(args.config)