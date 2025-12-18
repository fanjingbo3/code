# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
import sys
import copy
from tqdm import tqdm

# 引用项目模块
sys.path.append(os.getcwd())
from src.adaptive_quantum_cvrp.alm.classical_solver import ClassicalSolver
from src.adaptive_quantum_cvrp.alm.optimizer import ALMOptimizer
from src.adaptive_quantum_cvrp.rl.pearl_agent import PEARLAgent
from src.adaptive_quantum_cvrp.rl.meta_env import MetaCVRPEnv

# === 配置 ===
CHECKPOINT_PATH = "results/pearl_test_run/checkpoints/pearl_agent_950.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TEST_INSTANCES = 20  

def load_agent(env):
    agent = PEARLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        latent_dim=5,
        hidden_dim=256,
        device=DEVICE
    )
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'actor' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.context_encoder.load_state_dict(checkpoint['encoder'])
        else:
            agent.actor.load_state_dict(checkpoint)
        print(f"模型加载成功: {CHECKPOINT_PATH}")
    else:
        print(f"错误: 找不到模型 {CHECKPOINT_PATH}")
        sys.exit(1)
    return agent

def run_benchmark():
    dummy_solver = ClassicalSolver()
    env = MetaCVRPEnv(solver=dummy_solver)
    agent = load_agent(env)
    
    results = []
    
    print(f"开始批量测试 {NUM_TEST_INSTANCES} 道题目 (参数传递修正版)...")
    
    for i in tqdm(range(NUM_TEST_INSTANCES)):
        # 1. 生成一道新题
        task_id = np.random.randint(0, 4)
        env.reset_task(task_id)
        base_instance = env._generate_instance()
        
        # --- 方法 A: 传统 ALM (固定参数) ---
        instance_fixed = copy.deepcopy(base_instance)
        solver_fixed = ClassicalSolver()
        
        # 使用位置参数初始化 Optimizer
        opt_fixed = ALMOptimizer(instance_fixed, 50, solver_fixed)
        
        try:
            # 【关键修正！！！】必须使用 keyword argument，否则参数会被吞掉！
            res_fixed = opt_fixed.solve(initial_penalty_mu=10.0)
            cost_fixed = res_fixed['best_cost']
            feas_fixed = res_fixed['best_solution'] is not None
        except Exception:
            # 如果万一这里报错说 unexpected keyword，那就说明原项目定义太诡异了
            # 但根据 inference.py 能跑，这里必须加 keyword
            cost_fixed = 99999
            feas_fixed = False
        
        # --- 方法 B: 你的 PEARL (Meta-RL) ---
        env.current_instance = copy.deepcopy(base_instance)
        agent.clear_context()
        
        best_pearl_cost = float('inf')
        best_pearl_feas = False
        best_rho_used = 0
        
        # 试探 5 次
        for _ in range(5):
            obs = env.reset() 
            agent.infer_z()
            action = agent.select_action(obs, deterministic=False)
            
            raw_rho = action[0]
            rho = 10.0 ** (2 * raw_rho + 1)
            
            solver_rl = ClassicalSolver()
            instance_trial = copy.deepcopy(base_instance)
            instance_trial.name = f"{base_instance.name}_pearl_{_}_{np.random.randint(1000)}"
            opt_rl = ALMOptimizer(instance_trial, 50, solver_rl)
            try:
                # 【关键修正！！！】这里也要加 keyword argument
                res_rl = opt_rl.solve(initial_penalty_mu=rho)
                c = res_rl['best_cost']
                f = res_rl['best_solution'] is not None
                
                # Context 更新
                r_sim = -c / 50.0
                if not f: r_sim -= 20.0
                agent.update_context(obs, action, r_sim)

                if f and c < best_pearl_cost:
                    best_pearl_cost = c
                    best_pearl_feas = True
                    best_rho_used = rho
            except:
                pass
            
        # 记录结果
        p_cost = best_pearl_cost if best_pearl_feas else 99999
        f_cost = cost_fixed if feas_fixed else 99999
        
        # 计算 Gap
        if feas_fixed and best_pearl_feas:
            gap = (f_cost - p_cost) / f_cost * 100
        elif best_pearl_feas and not feas_fixed:
            gap = 100.0 
        elif not best_pearl_feas and feas_fixed:
            gap = -100.0
        else:
            gap = 0.0 
            
        results.append({
            "ID": i,
            "Type": env.active_task['type'],
            "N": env.active_task['n'],
            "Fixed_Cost": round(f_cost, 2),
            "PEARL_Cost": round(p_cost, 2),
            "Best_Rho": round(best_rho_used, 2),
            "Gap(%)": round(gap, 2)
        })

    # 输出表格
    df = pd.DataFrame(results)
    print("\n=== 测试结果摘要 ===")
    print(df.to_string())
    
    # 统计 (Gap > 0.01 算赢，排除浮点误差)
    wins = len(df[df["Gap(%)"] > 0.01])
    print(f"\nPEARL 优于 Fixed 的场次: {wins} / {NUM_TEST_INSTANCES}")
    print(f"平均提升幅度 (Gap): {df['Gap(%)'].mean():.2f}%")
    
    df.to_csv("benchmark_results.csv", index=False)
    print("结果已保存到 benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()