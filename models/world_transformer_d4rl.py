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
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.normalizer import StandardNormalizer
from common import util
from world_transformer import TimeSeriesTransformer

def set_global_device(dev):
    global device
    device = torch.device(dev)



class D4RLWorldModel:
    def __init__(self,
                 env_name,
                #  lr=1e-3,
                #  holdout_ratio=0.1,
                #  device='cuda:0',
                 dataset=None,
                 load_data = True,
                #  epochs = 50,
                #  hidden_dim=512,
                 args=None,
                 **kwargs):
        
        device = args.device
        self.env = gym.make(env_name)

        if load_data:
            if dataset is None:
                self.dataset = d4rl.qlearning_dataset(self.env)
            else:
                self.dataset = dataset
            print("loaded dataset")
        set_global_device(device)
        util.device = device


        self.epochs = args.epochs
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.device = device
        print(f"Using device: {self.device}")
   
        self.path = getattr(args, 'path', '/data/abiomed_tmp/processed')
        self.seq_dim = getattr(args, 'seq_dim', 12)
        self.output_dim = getattr(args, 'output_dim', 11*12)
        self.bc = getattr(args, 'bc', 64)
        self.nepochs = getattr(args, 'nepochs', 20)
        self.encs = getattr(args, 'encs', 1)
        self.lr = getattr(args, 'lr', 0.001)
        self.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)
        self.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
        self.dim_model = getattr(args, 'dim_model', 256)

        self.model = TimeSeriesTransformer(input_dim=self.obs_dim, output_dim=self.obs_dim, dim_model=self.dim_model,
                                                num_encoder_layers = self.encs, pl_shape = self.action_dim,
                                                encoder_dropout = self.encoder_dropout, 
                                                decoder_dropout = self.decoder_dropout, 
                                                device=self.device)
        # Initialize optimizers
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Initialize normalizers
        self.obs_normalizer = StandardNormalizer()
        self.act_normalizer = StandardNormalizer()
        
        # Training parameters
        self.holdout_ratio = args.holdout_ratio
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
                # pred_next_obs = self.model(torch.cat([batch_obs, batch_actions], dim=1))
                pred_next_obs = self.model(batch_obs.unsqueeze(1), batch_actions)
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

    def predict(self, obs_loader):

        batch = 0
        with torch.no_grad():
            all_outputs = []
            self.trained_model.eval()
            for src, pl, tgt in obs_loader: #why loader size 1
                outputs = []
                input_i = src
                for i in range(9):

                    pl_i = pl[:, i*10:(i+1)*10].to(self.device)
                    # print(pl_i.device)
                    # print(self.trained_model.device)
                    # print(input_i.device)
                    output = self.trained_model(input_i, pl_i)
                    output_reshaped = output.reshape([output.shape[0], 11, self.seq_dim])[:, 1:,:] #only take new predictions, ignore first datapoint
                    outputs.append(output_reshaped)
                    input_i = torch.concat([input_i[:,10:,:].to(self.device), output_reshaped], axis=1)
                #64x90x6
                pred = np.array(torch.concat(outputs, axis=1).detach().cpu())

 
    
def main(args, dataset=None):
    

    model = D4RLWorldModel(env_name=args.env_name, dataset = dataset, args=args)
    loss = model.train_model()
    print("finished training")
    if args.noisy:
        args.env_name = args.env_name + "_noisy"
    model.save_model(f"saved_models/{args.env_name}/world_model_{loss:.2f}.pth")
    print("saved model")
    
if __name__ == "__main__":
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="hopper-expert-v0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--noisy", action='store_true', help="whether to use noisy dataset")

    #model parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension of the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the model")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--holdout_ratio", type=float, default=0.1, help="holdout ratio for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")

    args = parser.parse_args()
    print()
    if args.data_path != "":
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = None
    
    main(args,data)
