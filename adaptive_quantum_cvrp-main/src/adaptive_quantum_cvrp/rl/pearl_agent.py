import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# 引用同目录下的 networks
from src.adaptive_quantum_cvrp.rl.pearl_networks import ContextEncoder, TanhGaussianPolicy, QNetwork

class PEARLAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        latent_dim,
        hidden_dim=256,  # <--- 【新增】接收 hidden_dim
        lr=3e-4,         # <--- 【新增】接收 learning_rate
        device="cuda"
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.alpha = 0.2
        self.gamma = 0.99
        self.kl_lambda = 0.05
        
        # 1. 初始化网络 (传入 hidden_dim)
        self.context_encoder = ContextEncoder(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.actor = TanhGaussianPolicy(state_dim, latent_dim, action_dim, hidden_dim).to(device)
        
        # Double Q-Learning
        self.q1 = QNetwork(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.target_q1 = QNetwork(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.target_q2 = QNetwork(state_dim, action_dim, latent_dim, hidden_dim).to(device)
        
        # 复制参数到 Target 网络
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # 优化器 (使用传入的 lr)
        self.encoder_optimizer = optim.Adam(self.context_encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        
        # 上下文存储
        self.context = None
        self.z = None

    def clear_context(self):
        self.context = None
        # 重置 z 为先验分布 (0向量或随机向量)
        self.z = torch.zeros(1, self.latent_dim).to(self.device)

    def update_context(self, state, action, reward):
        """收集当前 Task 的历史数据 (s, a, r)"""
        # 确保数据维度正确
        reward = np.array([reward]) if np.isscalar(reward) else reward
        
        o = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        r = torch.FloatTensor(reward).unsqueeze(0).to(self.device)
        
        new_transition = torch.cat([o, a, r], dim=-1)
        
        if self.context is None:
            self.context = new_transition
        else:
            self.context = torch.cat([self.context, new_transition], dim=0)

    def infer_z(self):
        """通过 Context Encoder 计算 z"""
        if self.context is None:
            self.z = torch.randn(1, self.latent_dim).to(self.device)
            return self.z
        
        # context shape: [history_len, dim]
        s = self.context[:, :6]  # 假设 state_dim 是 6，如果是别的维度这里需要动态改，或者在 encoder 内部处理
        # 为了通用性，最好是在 encoder 内部切分，这里简化处理，假设前6位是state
        # 更稳健的写法是传入 state_dim，这里先硬编码适配你的环境
        dim_s = 6 # 你的环境 observation 是 6 维
        dim_a = 2 # action 是 2 维
        
        s = self.context[:, :dim_s]
        a = self.context[:, dim_s:dim_s+dim_a]
        r = self.context[:, dim_s+dim_a:]
        
        mu, log_sigma = self.context_encoder(s, a, r)
        sigma = torch.exp(log_sigma)
        
        # Product of Gaussians (近似为均值的平均)
        z_mu = torch.mean(mu, dim=0, keepdim=True)
        z_sigma = torch.mean(sigma, dim=0, keepdim=True)
        
        dist = Normal(z_mu, z_sigma)
        self.z = dist.rsample()
        return self.z

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        z = self.z 
        
        if deterministic:
            action, _ = self.actor(state, z)
        else:
            action, _ = self.actor(state, z)
            
        return action.cpu().detach().numpy()[0]

    def train_step(self, meta_batch_size, context_batch, transition_batch):
        """
        元训练步骤
        context_batch: [meta_batch, context_len, dim]
        transition_batch: (s, a, r, s', d) tuple of tensors
        """
        # 1. Context Inference
        s_c = context_batch[..., :6]
        a_c = context_batch[..., 6:8]
        r_c = context_batch[..., 8:]
        
        mu, log_sigma = self.context_encoder(s_c, a_c, r_c)
        task_mu = mu.mean(dim=1)
        task_sigma = torch.exp(log_sigma).mean(dim=1)
        
        dist = Normal(task_mu, task_sigma)
        z = dist.rsample() # [meta_batch, latent_dim]
        
        # KL Divergence (Regularization)
        kl_div = -0.5 * (1 + 2 * torch.log(task_sigma) - task_mu.pow(2) - task_sigma.pow(2)).sum(dim=1).mean()
        
        # 2. Prepare SAC Data
        s, a, r, s_next, d = transition_batch
        
        # Expand z to match batch_size: [meta_batch, batch_size, latent]
        # transition_batch 的 shape 是 [meta_batch, batch_size, dim]
        batch_size = s.shape[1]
        z_expanded = z.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Flatten for network input
        s = s.view(-1, 6)
        a = a.view(-1, 2)
        r = r.view(-1, 1)
        s_next = s_next.view(-1, 6)
        d = d.view(-1, 1)
        z_flat = z_expanded.reshape(-1, self.latent_dim)
        
        # 3. Critic Update
        with torch.no_grad():
            next_action, next_log_prob = self.actor(s_next, z_flat)
            # 确保 next_log_prob 维度匹配 [batch, 1]
            if next_log_prob.dim() == 1:
                next_log_prob = next_log_prob.unsqueeze(1)
                
            target_q1 = self.target_q1(s_next, next_action, z_flat)
            target_q2 = self.target_q2(s_next, next_action, z_flat)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = r + self.gamma * (1 - d) * target_q
        
        current_q1 = self.q1(s, a, z_flat)
        current_q2 = self.q2(s, a, z_flat)
        q_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        # 4. Actor Update
        pred_action, log_prob = self.actor(s, z_flat)
        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(1)
            
        q1_pi = self.q1(s, pred_action, z_flat)
        q2_pi = self.q2(s, pred_action, z_flat)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()
        
        # 5. Backprop
        total_loss = actor_loss + q_loss + self.kl_lambda * kl_div
        
        self.encoder_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.actor_optimizer.step()
        self.q_optimizer.step()
        
        # Soft update targets
        tau = 0.005
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": q_loss.item(),
            "kl": kl_div.item()
        }
    
    # 增加 save 方法，防止 run_pearl.py 报错
    def save(self, path):
        torch.save({
            'encoder': self.context_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict()
        }, path)