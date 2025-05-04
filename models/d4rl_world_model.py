import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import d4rl
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.normalizer import StandardNormalizer
from common import util


class D4RLWorldModel:
    def __init__(self,
                 env_name,
                 lr=1e-3,
                 holdout_ratio=0.1,
                 device='cuda:0',
                 dataset=None,
                 load_data = True,
                 epochs = 50,
                 **kwargs):
        
        self.env = gym.make(env_name)

        if load_data:
            if dataset is None:
                self.dataset = d4rl.qlearning_dataset(self.env)
            else:
                self.dataset = dataset
            print("loaded dataset")
        util.set_global_device(device)

        self.epochs = epochs
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
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
        
        # make torch dataset and batch
        dataset = torch.utils.data.TensorDataset(obs, actions, next_obs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        # Train model
        l = 0
        self.model.train()
        for epoch in range(self.epochs):  # You can adjust number of epochs
            
            epoch_loss = []
            # use dataloader
            for batch_obs, batch_actions, batch_next_obs in dataloader:
                self.model_optimizer.zero_grad()

                # Forward pass
                pred_next_obs = self.model(torch.cat([batch_obs, batch_actions], dim=1))
                
                # Compute loss
                loss = nn.MSELoss()(pred_next_obs, batch_next_obs)
            
                # Backward pass
                loss.backward()
                self.model_optimizer.step()
                epoch_loss.append(loss.item())
            
            if epoch % 2 == 0:
                print(f'Epoch {epoch}, Loss: {np.mean(epoch_loss):.4f}')
                l = np.mean(epoch_loss)
        return l
         

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

    # def crps(self, x, y):
    #     #calculate crps for a single sample
    #     y_hat = self.model.predict_multiple(x)
    #     return self.model.crps(x, y)
    
    def save_model(self, path):
        """Save model and normalizers"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_normalizer': self.obs_normalizer,
            'act_normalizer': self.act_normalizer
        }, path)
        
    def load_model(self, path):
        """Load model and normalizers"""
        checkpoint = torch.load(path, map_location=f'{self.device}')
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
        input = torch.repeat(x, num_samples,1) #changed torch to numpy
        predictions = self.forward(input)
        return predictions.view(num_samples, -1)
    
    
def main(args):
    model = D4RLWorldModel(env_name=args.env_name, device=args.device, epochs=args.epochs)
    loss = model.train_model()
    print("finished training")
    model.save_model(f"saved_models/{args.env_name}/world_model_{loss:.2f}.pth")
    print("saved model")
    
if __name__ == "__main__":
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="halfcheetah-random-v0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    main(args)