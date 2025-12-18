# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import sys
from pathlib import Path

# 确保能引用到 src 目录
sys.path.append(os.getcwd())

from src.adaptive_quantum_cvrp.common.cvrp_instance import CVRPInstance
from src.adaptive_quantum_cvrp.alm.classical_solver import ClassicalSolver
from src.adaptive_quantum_cvrp.rl.pearl_agent import PEARLAgent
from src.adaptive_quantum_cvrp.rl.meta_env import MetaCVRPEnv

# === 配置 ===
# 指向你刚才生成的模型文件 (确保文件名正确)
CHECKPOINT_PATH = "results/pearl_test_run/checkpoints/pearl_agent_950.pth" 
# 如果没有真实数据，设为 False，会自动生成一个测试题
USE_REAL_FILE = False 
REAL_FILE_PATH = "data/cvrplib_instances/A-n32-k5.vrp" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_agent(path, env):
    # 初始化一个空 Agent
    agent = PEARLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        latent_dim=5,
        hidden_dim=256,
        device=DEVICE
    )
    
    # 加载权重
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE)
        
        if 'actor' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.context_encoder.load_state_dict(checkpoint['encoder'])
            print(f"成功加载模型: {path}")
        else:
            print("警告：模型格式可能不匹配，尝试直接加载...")
            try:
                agent.actor.load_state_dict(checkpoint)
            except:
                print("加载失败，请检查 save 逻辑")
    else:
        print(f"错误：找不到模型文件 -> {path}")
        return None
        
    return agent

def run_inference():
    # 1. 准备环境
    solver = ClassicalSolver()
    env = MetaCVRPEnv(solver=solver)
    
    # 2. 准备题目
    if USE_REAL_FILE and os.path.exists(REAL_FILE_PATH):
        print(f"正在读取真实文件: {REAL_FILE_PATH}")
        instance = CVRPInstance.from_file(REAL_FILE_PATH)
    else:
        print("生成一个随机测试题 (N=32, Clustered)...")
        env.reset_task(1) # 聚类任务
        env.active_task['n'] = 32
        env.active_task['capacity'] = 100
        instance = env._generate_instance()

    # 3. 加载 AI
    agent = load_agent(CHECKPOINT_PATH, env)
    if agent is None:
        return

    # 4. PEARL 推理流程 (Context Adaptation)
    print("\n=== 开始 Meta-RL 自适应推理 ===")
    env.current_instance = instance
    
    # Step A: 试探 (Exploration)
    agent.clear_context()
    print("正在试探题目特征 (Adaptation)...")
    
    # 记录历史最佳
    best_cost = float('inf')
    best_action_recorded = None
    best_result_info = None
    
    for i in range(5):
        # reset 只是为了刷新 obs 计算
        obs = env.reset() 
        agent.infer_z()
        
        # 这里的关键是 deterministic=False
        # 我们利用 SAC 的随机性去探索最优参数区间
        action = agent.select_action(obs, deterministic=False)
        
        # 执行 ALM
        _, r, _, info = env.step(action)
        
        # 更新上下文
        agent.update_context(obs, action, r)
        
        cost = info['cost']
        
        # === 核心逻辑：记录历史最佳 ===
        if cost < best_cost:
            best_cost = cost
            best_action_recorded = action
            best_result_info = info
            
        print(f"  [试探 {i+1}] Cost: {cost:.2f} (参数: {action[0]:.2f}, {action[1]:.2f})")

    # Step B: 结算
    # 我们不再跑第 6 次“执行”了，直接汇报前 5 次里最好的那个结果
    # 这就是 "Best-of-History" 策略
    
    print("-" * 40)
    print(f"题目规模: N={instance.num_customers}")
    print(f"最佳 Cost:         {best_cost:.2f}")
    
    if best_action_recorded is not None:
        # 手动还原一下真实参数，方便你看
        raw_rho, raw_sigma = best_action_recorded[0], best_action_recorded[1]
        real_rho = 10.0 ** (2 * raw_rho + 1)
        real_sigma = 10.0 ** (2 * raw_sigma + 1)
        print(f"最佳参数 (Real):   rho={real_rho:.4f}, sigma={real_sigma:.4f}")
        print(f"是否可行解:        {best_result_info['feasible']}")
    
    print("-" * 40)

if __name__ == "__main__":
    run_inference()