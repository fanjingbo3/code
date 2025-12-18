import argparse
import logging
from pathlib import Path
import json
import numpy as np
import torch

# --- 使用全路径引用 ---
from src.adaptive_quantum_cvrp.alm.classical_solver import ClassicalSolver
from src.adaptive_quantum_cvrp.rl.pearl_agent import PEARLAgent
from src.adaptive_quantum_cvrp.rl.meta_env import MetaCVRPEnv
from src.adaptive_quantum_cvrp.rl.replay_buffer import MetaReplayBuffer
from src.adaptive_quantum_cvrp.utils.config_loader import load_config
from src.adaptive_quantum_cvrp.utils.logging_config import setup_logging

# 检测 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pearl_meta_training(config, solver, base_output_dir):
    """
    PEARL 元训练循环 (运行在 GPU 上)
    """
    logging.info(f"Starting PEARL Meta-Training on device: {DEVICE}")
    
    pearl_config = config["pearl"]
    meta_env = MetaCVRPEnv(solver=solver) 
    
    # 1. 初始化 Agent
    agent = PEARLAgent(
        state_dim=meta_env.observation_space.shape[0],
        action_dim=meta_env.action_space.shape[0],
        latent_dim=pearl_config["latent_dim"],
        hidden_dim=pearl_config["hidden_dim"],
        lr=pearl_config["learning_rate"],
        device=DEVICE
    )

    # 2. 初始化 Replay Buffer (这里是你报错的地方，现在修复了)
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
        # 采样任务 ID
        task_indices = np.random.choice(len(meta_env.tasks), meta_batch_size)
        
        # --- 数据收集阶段 ---
        for idx in task_indices:
            meta_env.reset_task(idx)
            agent.clear_context() # 清除上下文
            
            # Step A: 积累 Context (试探性运行)
            for _ in range(num_context_episodes):
                obs = meta_env.reset()
                agent.infer_z() # 推断 z
                
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, done, _ = meta_env.step(action)
                
                agent.update_context(obs, action, reward)
            
            # Step B: 收集训练数据 (基于已知的 Context)
            agent.infer_z() 
            for _ in range(num_train_episodes):
                obs = meta_env.reset()
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, done, _ = meta_env.step(action)
                
                # 存入 Buffer
                replay_buffer.add(task_idx=idx, 
                                  state=obs, action=action, reward=reward, 
                                  next_state=next_obs, done=done)

        # --- 模型更新阶段 ---
        loss_info = {}
        for _ in range(pearl_config["num_gradient_steps"]):
            # 从 Buffer 采样 Batch
            context_batch, transition_batch = replay_buffer.sample_batch(meta_batch_size)
            # 训练一步
            loss_dict = agent.train_step(meta_batch_size, context_batch, transition_batch)
            loss_info.update(loss_dict)

        # 打印日志
        if iteration % 10 == 0:
            logging.info(f"Iter {iteration}/{num_iterations} | "
                         f"Actor Loss: {loss_info.get('actor_loss', 0):.4f} | "
                         f"Critic Loss: {loss_info.get('critic_loss', 0):.4f} | "
                         f"KL: {loss_info.get('kl', 0):.4f}")
            
        # 保存模型
        # 保存模型
        if iteration % 50 == 0:
            save_path = base_output_dir / "checkpoints"
            save_path.mkdir(parents=True, exist_ok=True)
            # 取消注释，真正执行保存
            agent.save(save_path / f"pearl_agent_{iteration}.pth")
            logging.info(f"Model saved to {save_path / f'pearl_agent_{iteration}.pth'}")

    # --- 循环结束后，再保存一次最终模型 ---
    final_save_path = base_output_dir / "checkpoints" / "pearl_agent_final.pth"
    agent.save(final_save_path)
    logging.info(f"Final model saved to {final_save_path}")

    logging.info("PEARL Meta-Training Complete.")
    return agent

    logging.info("PEARL Meta-Training Complete.")
    return agent

def run(config_path: Path):
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    base_output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(base_output_dir, config["experiment"]["log_level"])
    logging.info(f"Loaded configuration. GPU Available: {torch.cuda.is_available()}")

    # 强制使用经典求解器
    solver = ClassicalSolver()
    
    # 开始训练
    run_pearl_meta_training(config, solver, base_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)