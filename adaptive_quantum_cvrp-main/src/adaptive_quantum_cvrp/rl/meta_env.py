import numpy as np
import math

# 引用你项目真实的 Instance 类
from src.adaptive_quantum_cvrp.common.cvrp_instance import CVRPInstance
from src.adaptive_quantum_cvrp.alm.classical_solver import ClassicalSolver
# 引用 Optimizer 用于 step 中计算 reward
from src.adaptive_quantum_cvrp.alm.optimizer import ALMOptimizer

class MetaCVRPEnv:
    def __init__(self, solver=None):
        # 定义任务：不同分布、不同规模
        self.tasks = [
            {'id': 0, 'type': 'random', 'n': 20, 'capacity': 30},
            {'id': 1, 'type': 'clustered', 'n': 20, 'capacity': 30},
            {'id': 2, 'type': 'random', 'n': 50, 'capacity': 100},
            {'id': 3, 'type': 'clustered', 'n': 50, 'capacity': 100},
        ]
        self.active_task = self.tasks[0]
        # 这里的 solver 传入是实例化好的对象
        self.solver = solver if solver else ClassicalSolver()
        
        # 状态空间维度 (ALM 特征)
        self.observation_space = type('obj', (object,), {'shape': (6,)})
        # 动作空间维度 (rho, sigma)
        self.action_space = type('obj', (object,), {'shape': (2,)})

    def reset_task(self, idx):
        self.active_task = self.tasks[idx]

    def _generate_instance(self):
        """生成符合 CVRPInstance 定义的实例"""
        n = self.active_task['n']
        cap = self.active_task['capacity']
        
        # 1. 生成数据 (List 格式)
        coords = [np.random.uniform(0, 100, 2).tolist()] # Depot
        demands = [0]
        
        if self.active_task['type'] == 'random':
            for _ in range(n):
                coords.append(np.random.uniform(0, 100, 2).tolist())
                demands.append(int(np.random.randint(1, 10)))
        else:
            # 聚类分布
            centers = np.random.uniform(0, 100, (3, 2))
            for _ in range(n):
                center = centers[np.random.randint(0, 3)]
                noise = np.random.normal(0, 5, 2)
                pos = np.clip(center + noise, 0, 100)
                coords.append(pos.tolist())
                demands.append(int(np.random.randint(1, 10)))

        # 2. 转换为 Numpy
        nodes_np = np.array(coords)
        demands_np = np.array(demands)
        
        # 3. 计算统计量
        num_customers = len(coords) - 1
        total_demand = np.sum(demands_np)
        min_vehicles = math.ceil(total_demand / cap)
        num_vehicles = min_vehicles + 2 
        
        # 4. 实例化
        instance = CVRPInstance(
            name=f"gen_{self.active_task['type']}_{n}",
            num_customers=num_customers,
            capacity=cap,
            demands=demands_np,
            num_vehicles=num_vehicles,
            nodes=nodes_np
        )
        return instance

    def reset(self):
        self.current_instance = self._generate_instance()
        
        # 提取初始特征
        dists = self.current_instance.dist_matrix.flatten()
        demands = self.current_instance.demands
        
        obs = np.array([
            self.current_instance.num_customers,
            self.current_instance.capacity,
            np.mean(dists),
            np.std(dists),
            np.mean(demands),
            np.std(demands)
        ])
        
        obs = np.log1p(obs) 
        return obs

    def step(self, action):
        """
        核心步骤函数
        """
        # 1. 参数映射 (解决负数参数问题)
        # raw_rho, raw_sigma 在 [-1, 1] 之间
        raw_rho, raw_sigma = action[0], action[1]

        # 映射到 10^0.1 (1.2) ~ 10^3 (1000) 附近
        # 2 * raw + 1 的范围是 [-1, 3]
        # 10 ** [-1, 3] -> [0.1, 1000]
        rho = 10.0 ** (2 * raw_rho + 1)
        sigma = 10.0 ** (2 * raw_sigma + 1)

        # 2. 实例化求解器
        # 增加 max_iterations 到 50 以保证效果
        # 使用位置参数避免 kwarg 报错
        optimizer = ALMOptimizer(self.current_instance, 50, self.solver)
        
        try:
            # 3. 执行求解
            # 传入映射后的正数参数
            # 如果 solver.solve 只支持一个参数，这里只传 rho
            result = optimizer.solve(initial_penalty_mu=rho) 
            
            cost = result['best_cost']
            is_feasible = result['best_solution'] is not None
        except Exception as e:
            cost = 100000.0
            is_feasible = False
            
        # 4. Reward 计算 (解决 Loss 震荡问题)
        # 缩放 Cost
        scaled_cost = cost / 50.0
        
        reward = -scaled_cost
        if not is_feasible:
            # 惩罚改为一个温和的值
            reward -= 20.0 
        
        done = True 
        next_obs = np.zeros(6) 
        
        return next_obs, reward, done, {'cost': cost, 'feasible': is_feasible}