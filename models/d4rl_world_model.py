import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import d4rl
from common.normalizer import StandardNormalizer
from common import util

class D4RLWorldModel:
    def __init__(self,
                 env_name,
                 obs_space,
                 action_space,
                 lr=1e-3,
                 holdout_ratio=0.1,
                 device='cuda',
                 **kwargs):
        
        self.env = gym.make(env_name)
        self.dataset = d4rl.qlearning_dataset(self.env)
        
        self.obs_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.device = device
        
        # Initialize model
        self.model = MLPNetwork(obs_dim=self.obs_dim, 
                              action_dim=self.action_dim, 
                              device=self.device)
        
        # Initialize optimizers
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize normalizers
        self.obs_normalizer = StandardNormalizer()
        self.act_normalizer = StandardNormalizer()
        
        # Training parameters
        self.holdout_ratio = holdout_ratio
        self.model_train_timesteps = 0
        
    def train_model(self, data=None):
        if data is None:
            data = self.dataset
            
        # Split into train and validation sets
        n = len(data['observations'])
        train_n = int(n * (1 - self.holdout_ratio))
        
        # Normalize data
        obs = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        next_obs = torch.FloatTensor(data['next_observations']).to(self.device)
        
        # Update normalizers
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(actions)
        
        # Transform data
        obs = self.obs_normalizer.transform(obs)
        actions = self.act_normalizer.transform(actions)
        
        # Train model
        self.model.train()
        for epoch in range(100):  # You can adjust number of epochs
            self.model_optimizer.zero_grad()
            
            # Forward pass
            pred_next_obs = self.model(torch.cat([obs, actions], dim=1))
            
            # Compute loss
            loss = nn.MSELoss()(pred_next_obs, next_obs)
            
            # Backward pass
            loss.backward()
            self.model_optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
                
        return loss.item()
    
    def predict(self, obs, action):
        """Predict next state given current state and action"""
        self.model.eval()
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            
            # Normalize inputs
            obs = self.obs_normalizer.transform(obs)
            action = self.act_normalizer.transform(action)
            
            # Predict
            pred_next_obs = self.model(torch.cat([obs, action], dim=1))
            
            # Denormalize output
            pred_next_obs = self.obs_normalizer.inverse_transform(pred_next_obs)
            
        return pred_next_obs.cpu().numpy()
    
    def save_model(self, path):
        """Save model and normalizers"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_normalizer': self.obs_normalizer,
            'act_normalizer': self.act_normalizer
        }, path)
        
    def load_model(self, path):
        """Load model and normalizers"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.obs_normalizer = checkpoint['obs_normalizer']
        self.act_normalizer = checkpoint['act_normalizer']


class MLPNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, device='cuda', hidden_dim=256, dropout=0.1):
        super(MLPNetwork, self).__init__()
        
        self.input_dim = obs_dim + action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, obs_dim)
        ).to(device)
        
    def forward(self, x):
        return self.network(x)
    
    @torch.no_grad()
    def predict_multiple(self, x, num_samples=10):
        predictions = []
        for _ in range(num_samples):
            predictions.append(self.forward(x))
        return torch.stack(predictions)
    
    
    