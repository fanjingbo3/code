import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np  # <--- 之前漏了这行

class ContextEncoder(nn.Module):
    """
    接收 (state, action, reward) -> 输出高斯分布的 mu 和 sigma -> 采样得到 z
    """
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # 输入是 state + action + reward (1维)
        self.input_dim = state_dim + action_dim + 1
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state, action, reward):
        # 拼接上下文信息
        x = torch.cat([state, action, reward], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x)
        # 限制 sigma 范围，防止数值不稳定
        log_sigma = torch.clamp(log_sigma, -10, 2) 
        return mu, log_sigma

class TanhGaussianPolicy(nn.Module):
    """
    接收 state 和 z (Latent Context) -> 输出动作 (rho, sigma)
    """
    def __init__(self, state_dim, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 输入增加了 latent_dim
        self.fc1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        
        # 重参数化采样 (Reparameterization Trick)
        action_raw = dist.rsample() 
        action = torch.tanh(action_raw)
        
        # 计算 Log Probability (用于 SAC Loss)
        # 修正：确保这里能调用 np.log
        log_prob = dist.log_prob(action_raw).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - action_raw - F.softplus(-2 * action_raw))).sum(axis=-1)
        
        return action, log_prob

class QNetwork(nn.Module):
    """
    Critic Network: 接收 (state, action, z) -> 输出 Q 值
    """
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # 输入增加了 latent_dim
        self.fc1 = nn.Linear(state_dim + action_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, z):
        x = torch.cat([state, action, z], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)