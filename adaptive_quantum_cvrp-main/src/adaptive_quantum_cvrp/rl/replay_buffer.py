import numpy as np
import torch

class MetaReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, latent_dim, device):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # 存储数据
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))
        self.task_ids = np.zeros((max_size, 1), dtype=int)

    def add(self, task_idx, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.task_ids[self.ptr] = task_idx
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, meta_batch_size):
        """
        采样两个 Batch：
        1. Context Batch: 固定长度 (如 64)
        2. Transition Batch: 固定长度 (如 256)
        """
        available_tasks = np.unique(self.task_ids[:self.size])
        
        # 如果还没跑够 meta_batch_size 个任务，就允许重复采样任务
        if len(available_tasks) < meta_batch_size:
            batch_tasks = np.random.choice(available_tasks, meta_batch_size, replace=True)
        else:
            batch_tasks = np.random.choice(available_tasks, meta_batch_size, replace=False)
            
        context_list = []
        transition_list = []
        
        # 设定固定的采样长度，防止报错
        CONTEXT_SIZE = 64  # <--- 强制每个 Task 采 64 条 Context
        TRANS_SIZE = 256   # <--- 强制每个 Task 采 256 条 Transition
        
        for task_idx in batch_tasks:
            indices = np.where(self.task_ids[:self.size] == task_idx)[0]
            
            # --- 修复点 1: Context 采样 ---
            # 无论 indices 有多少个，都强制采 CONTEXT_SIZE 个
            # replace=True 允许重复，这样如果数据只有 10 条，它会重复采样凑满 64 条
            if len(indices) > 0:
                context_indices = np.random.choice(indices, size=CONTEXT_SIZE, replace=True)
            else:
                # 理论上不该进这里，防止万一
                continue

            s = torch.tensor(self.states[context_indices], dtype=torch.float32).to(self.device)
            a = torch.tensor(self.actions[context_indices], dtype=torch.float32).to(self.device)
            r = torch.tensor(self.rewards[context_indices], dtype=torch.float32).to(self.device)
            context_list.append(torch.cat([s, a, r], dim=-1))
            
            # --- 修复点 2: Transition 采样 ---
            if len(indices) > 0:
                trans_indices = np.random.choice(indices, size=TRANS_SIZE, replace=True)
            else:
                continue

            transition_list.append((
                torch.tensor(self.states[trans_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.actions[trans_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.rewards[trans_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.next_states[trans_indices], dtype=torch.float32).to(self.device),
                torch.tensor(self.dones[trans_indices], dtype=torch.float32).to(self.device)
            ))
            
        # Stack 要求 List 里每个 Tensor 形状完全一致
        # 现在每个都是 [CONTEXT_SIZE, dim]，所以 stack 不会报错了
        context_batch = torch.stack(context_list)
        
        s_batch = torch.stack([t[0] for t in transition_list])
        a_batch = torch.stack([t[1] for t in transition_list])
        r_batch = torch.stack([t[2] for t in transition_list])
        ns_batch = torch.stack([t[3] for t in transition_list])
        d_batch = torch.stack([t[4] for t in transition_list])
        
        return context_batch, (s_batch, a_batch, r_batch, ns_batch, d_batch)