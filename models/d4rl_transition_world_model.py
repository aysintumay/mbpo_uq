import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import d4rl
import argparse
import pickle
import sys
import os
import importlib
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.normalizer import StandardNormalizer
from common import util

def set_global_device(dev):
    global device
    device = torch.device(dev)

class D4RLWorldModel:
    def __init__(self,
                 env_name,
                 lr=1e-3, #increase
                 holdout_ratio=0.1,
                 device='cuda:0',
                 dataset=None,
                 load_data = True,
                 epochs = 50,
                 hidden_dim=512,
                 lambda_obs = 0.8,
                #  args=None,
                 **kwargs):
        
        # device = args.device
        self.env = gym.make(env_name)

        if load_data:
            if dataset is None:
                self.dataset = d4rl.qlearning_dataset(self.env)
            else:
                self.dataset = dataset
            print("loaded dataset")
        set_global_device(device)
        util.device = device

        self.scale = 0.1
        self.epochs = epochs
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = device
        print(f"Using device: {self.device}")
        
        self.lambda_obs = lambda_obs
        # Initialize model
        self.model = MLPNetwork(obs_dim=self.obs_dim, 
                              action_dim=self.action_dim, 
                              hidden_dim=hidden_dim,
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
        obs = torch.FloatTensor(data['observations'][:train_n]).to(self.device)
        actions = torch.FloatTensor(data['actions'][:train_n]).to(self.device)
        next_obs = torch.FloatTensor(data['next_observations'][:train_n]).to(self.device)
        rewards = torch.FloatTensor(data['rewards'][:train_n]).to(self.device)

        val_obs = torch.FloatTensor(data['observations'][train_n:]).to(self.device)
        val_actions = torch.FloatTensor(data['actions'][train_n:]).to(self.device)
        val_next_obs = torch.FloatTensor(data['next_observations'][train_n:]).to(self.device)
        val_rewards = torch.FloatTensor(data['rewards'][train_n:]).to(self.device)

    
        self.obs_normalizer.update(obs)
        self.act_normalizer.update(actions)
        
        # Transform data
        # obs = self.obs_normalizer.transform(obs)
        # actions = self.act_normalizer.transform(actions)
        # next_obs = self.obs_normalizer.transform(next_obs)
       
        # make torch dataset and ba
        # tch
        dataset = torch.utils.data.TensorDataset(obs, actions, next_obs, rewards)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(val_obs, val_actions, val_next_obs, val_rewards)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Train model
        l = 0
        self.model.train()
        for epoch in range(self.epochs):  
            
            epoch_loss = []
            # use dataloader
            for batch_obs, batch_actions, batch_next_obs, true_reward in dataloader:
                self.model_optimizer.zero_grad()

                # Forward pass
                pred_next_obs, pred_reward = self.model(torch.cat([batch_obs, batch_actions], dim=1))

                obs_loss = nn.MSELoss()(pred_next_obs, batch_next_obs)
                reward_loss = nn.MSELoss()(pred_reward, true_reward)
                loss = self.lambda_obs * obs_loss + (1 - self.lambda_obs) * reward_loss
                # Compute loss
                # loss = nn.MSELoss()(pred_next_obs, batch_next_obs)
            
                # Backward pass
                loss.backward()
                self.model_optimizer.step()
                epoch_loss.append(loss.item())
            
            if epoch % 2 == 0:
                print(f'Epoch {epoch}, Obs Loss: {obs_loss.item():.4f}, Reward Loss: {reward_loss.item():.4f}, Loss: {np.mean(epoch_loss):.4f}')
                l = np.mean(epoch_loss)
                # Validate model
                self.model.eval()

                val_loss = []
                with torch.no_grad():
                    for val_batch_obs, val_batch_actions, val_batch_next_obs, val_true_reward in val_dataloader:
                        pred_val_next_obs, pred_val_reward = self.model(torch.cat([val_batch_obs, val_batch_actions], dim=1))
                        val_obs_loss = nn.MSELoss()(pred_val_next_obs, val_batch_next_obs)
                        val_reward_loss = nn.MSELoss()(pred_val_reward, val_true_reward)
                        val_loss.append(self.lambda_obs * val_obs_loss + (1 - self.lambda_obs) * val_reward_loss)
                val_loss = torch.mean(torch.stack(val_loss)).item()
                print(f'Validation Loss: {val_loss:.4f}')
                print("Val reward loss: ", val_reward_loss.item(), "Val obs loss: ", val_obs_loss.item())

            
        return l
         

    def predict(self, obs, action, uq=None, num_samples=50):
        """Predict next state given current state and action"""
        import_path = f"static_fns.hopper"
        static_fns = importlib.import_module(import_path).StaticFns
        self.model.eval()
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            if uq is None:
                # Normalize inputs
                # obs = self.obs_normalizer.transform(obs)
                # action = self.act_normalizer.transform(action)
                # Predict
                print('no uq')
                pred_next_obs, pred_reward = self.model(torch.cat([obs, action], dim=1))
                # Denormalize output
                # pred_next_obs = self.obs_normalizer.inverse_transform(pred_next_obs)
                
                
            else:

                # TODO: make it functional
                print("uq")
                states = np.repeat(obs.detach().cpu().numpy(), num_samples, axis=0)
                actions = np.repeat(action.detach().cpu().numpy(), num_samples, axis=0)
                print("states shape: ", states.shape)

                pred_input = torch.FloatTensor(np.concatenate([states, actions],axis = 1)).to(self.device)

                next_obs, next_reward = self.model(pred_input)
                next_obs_samples, next_reward_samples  = self.model.predict_multiple(pred_input, num_samples=num_samples)
                # Reshape to [batch_size, num_samples, obs_dim]
                obs_dim = obs.shape[-1]
                next_obs_samples = next_obs_samples.reshape(1, num_samples, obs_dim)
                next_reward_samples = next_reward_samples.reshape(1, num_samples, 1)
                
                # # next_obs_means = world_model.predict(obs_batch, action_batch)  #if use the next_obs_batch instead of the predicted next_obs

                # # Calculate standard deviation across samples for each batch item
                batch_stds = np.std(next_obs_samples, axis=1).max(axis=1)
                batch_stds_rew = np.std(next_reward_samples, axis=1).mean(axis=1, keepdims=True)

                penalized_rewards = next_reward - self.scale * batch_stds
                print("rewards shape: ", next_reward.mean())
                print("penalized rewards shape: ", batch_stds.mean())
                print(next_reward.shape)
                print(batch_stds.shape)

                # penalized_rewards = reward_batch
                pred_reward = penalized_rewards
                pred_next_obs = next_obs_samples.mean(axis=1)
            terminals = static_fns.termination_fn(obs.detach().cpu().numpy(), action.detach().cpu().numpy(), pred_next_obs.detach().cpu().numpy())
            terminals = terminals[:, None]
        return pred_next_obs.cpu().numpy(), pred_reward.cpu().numpy().reshape(-1,1), terminals, None

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
    def __init__(self, obs_dim, action_dim, device='cuda', hidden_dim=512, dropout=0.1):
        super(MLPNetwork, self).__init__()
        
        self.input_dim = obs_dim + action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.network1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, obs_dim)
        ).to(device)
        self.network2= nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(self.hidden_dim, obs_dim)
        ).to(device)

        self.obs_head = nn.Linear(hidden_dim, obs_dim).to(device)
        self.rew_head = nn.Linear(hidden_dim, 1).to(device)  
    def forward(self, x):
        h1 = self.network1(x)
        h2 = self.network2(x)

        next_obs = self.obs_head(h1)
        reward = self.rew_head(h2)
        return next_obs, reward.squeeze(-1)
    
    @torch.no_grad()
    def predict_multiple(self, x, num_samples=10):
        input = np.repeat(x, num_samples,1) #changed torch to numpy
        predictions, reward_preds = self.forward(input)
        return predictions, reward_preds 
    
    
def main(args, dataset=None):
    

    model = D4RLWorldModel(env_name=args.env_name, dataset = dataset, device=args.device, epochs=args.epochs)
    loss = model.train_model()
    print("finished training")
    if args.noisy:
        args.env_name = args.env_name + "_noisy"
    model.save_model(f"saved_models/{args.env_name}/transition_world_model_v2_{loss:.2f}_{args.n}.pth")
    print("saved model")
    
if __name__ == "__main__":
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="hopper-expert-v0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--n", type=float, default=0.00)

    
    #/abiomed/intermediate_data_d4rl/hopper-expert-v0_noisy_0.1.pkl
    #model parameters
    # parser.add_argument("--hidden_dim", type=int, default=512, help="hidden dimension of the model")
    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the model")
    # parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    # parser.add_argument("--holdout_ratio", type=float, default=0.1, help="holdout ratio for training")
    # parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args = parser.parse_args()
    args.noisy = True
    
 
    print()
    if args.data_path != "":
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = None
    
    main(args, data)
