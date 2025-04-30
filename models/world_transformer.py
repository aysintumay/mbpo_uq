import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join, abspath
import time
import os
import sys
import argparse
import logging

from common import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class WorldTransformer:
    def __init__(self, args, logger, pretrained = True):
        super(WorldTransformer, self).__init__()

        self.path = getattr(args, 'path', '/data/abiomed_tmp/processed')
        self.seq_dim = getattr(args, 'seq_dim', 12)
        self.output_dim = getattr(args, 'output_dim', 11*12)
        self.bc = getattr(args, 'bc', 64)
        self.nepochs = getattr(args, 'nepochs', 20)
        self.encs = getattr(args, 'encs', 2)
        self.lr = getattr(args, 'lr', 0.001)
        self.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)
        self.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
        self.dim_model = getattr(args, 'dim_model', 256)
        self.args = args

        
        self.logger = logger
        self.device = util.device

        self.model = TimeSeriesTransformer(input_dim=self.seq_dim, output_dim=self.output_dim, dim_model=self.dim_model,
                                                num_encoder_layers = self.encs, pl_shape = 10,
                                                encoder_dropout = self.encoder_dropout, 
                                                decoder_dropout = self.decoder_dropout, 
                                                device=self.device)
        # self.train_loader = self.read_data('train')
        # self.test_loader = self.read_data(mode = 'test') 
        self.model_save_dir = util.logger_model.log_path
       
        if pretrained:
            self.trained_model = self.load_model()
            print('loaded model')
        else:
            if args.data_name == 'train':   
                self.train_loader = self.read_data('train')
            else:
                self.test_loader = self.read_data('test')
            self.trained_model = self.train_model()
            print('trained model')
        self.rwd_mean = None
        self.rwd_std = None


    def train_model(self):

        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        train_losses = []
        print(f"number of epochs {self.args.nepochs}")
        for epoch in range(self.args.nepochs):
            self.model.train()
            total_loss = 0
            for src, pl, tgt in tqdm(self.train_loader):
                src = src.to(self.device)
                pl = pl.to(self.device)
                optimizer.zero_grad()
                output = self.model(src, pl)
                loss = criterion(output, tgt.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
            train_losses.append(avg_loss)

        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        torch.save(self.model.to('cpu').state_dict(),  os.path.join('/data/models/world_model', f"checkpoint_epoch_v_1_{self.args.task}_{self.nepochs}.pth"))
        self.model.to(self.device)
        self.logger.print("World model total time: {:.3f}s".format(time.time() - start_time))
        return self.model


    def read_data(self, mode='train'):

        dta = torch.load(os.path.join(self.path, 'pp_train_amicgs.pt')).numpy()
        dta = dta[: ,:, :-1]
        self.rwd_mean = dta.mean(axis=(0, 1))
        self.rwd_std = dta.std(axis=(0, 1))
        horizon = int(self.output_dim/self.seq_dim-1)
        if mode == 'test':
            dta = torch.load(os.path.join(self.path, 'pp_test_amicgs.pt')).numpy()
            dta = dta[: ,:, :-1]
            horizon = 90

        x_n = (dta - self.rwd_mean) / self.rwd_std
        x, pl, y = self.prep_transformer_world(x_n, ts= horizon+1, dims = 12)
        print("plshape is ", pl.shape)
        pl = pl[..., :horizon]
        dataset = TimeSeriesDataset(x, pl, y)
        loader = DataLoader(dataset, batch_size=self.bc, shuffle=True)
        return loader

    def resize(self, obs, action, next_state):
        # resize [N,1080] to [N,90, 12]
        n = int(obs.shape[0]/(12*90))
        #dont take the p-level into the observation space
        x = obs.reshape(n,90,12)
        y = next_state.reshape((n, 90*12))
        action = action.reshape(-1,90)
        loader = DataLoader(TimeSeriesDataset(x, action, y), batch_size=self.args.bc, shuffle=False)
        return loader
    
    def load_model(self):
        # Load the model state dict
   
        # self.model.load_state_dict(torch.load(os.path.join(self.logger.writer.get_logdir(), f"checkpoint_epoch_{self.args.epoch}.pth")))
        self.model.load_state_dict(torch.load(os.path.join('/data/models/world_model', f"checkpoint_epoch_v_1_{self.args.task}_{self.nepochs}.pth")))
        return self.model.to(self.device)
    
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
                # all_outputs.append(pred.detach().cpu())

        # final = np.array(all_outputs).reshape(-1,90,12)

        #calculate the MSE loss and print
        return pred  

    def prep_transformer_world(self, x_n, ts, dims=6):
            
            n = x_n.shape[0]
            #dont take the p-level into the observation space
            x = x_n[:,:90,:-1]
            y = x_n[:,89:89+ts,:-1].reshape((n, ts*dims))
            pl = x_n[:,90:,-1]
            return x, pl, y
    



class Decoder(nn.Module):
    def __init__(self, input_size=82, hidden_size=512, output_size=66, dropout=0):
        #output of the encoder+controlled variable (p-level)
        super(Decoder, self).__init__()
        # print(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TimeSeriesDataset(Dataset):
    def __init__(self, data, pl, labels):
        self.data = data
        self.pl = pl
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.pl[idx], self.labels[idx]
    

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model=512, num_heads=8, num_encoder_layers=3, encoder_dropout=0.1, 
                        decoder_dropout=0, max_len = 100, pl_shape=10, device='cpu'):
        super(TimeSeriesTransformer, self).__init__()
        
        self.device = torch.device(device)

        self.dim_model = dim_model
        self.input_embedding = nn.Linear(input_dim, dim_model).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=encoder_dropout).to(self.device),
            num_layers=num_encoder_layers
        ).to(self.device)
        self.dim_model = dim_model
        
        self.positional_encoding = self.create_positional_encoding(max_len, dim_model).to(self.device)

        self.decoder = Decoder(input_size = dim_model+pl_shape, output_size = output_dim, dropout = decoder_dropout).to(self.device)
    
    def create_positional_encoding(self, max_len, dim_model):
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, pl):
        src = self.input_embedding(src.to(self.device)) * np.sqrt(self.dim_model)
        #src = self.input_projection(src) * np.sqrt(self.dim_model)

        src += self.positional_encoding[:, :src.size(1)].clone().detach()
        
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        encoded_src = self.transformer_encoder(src)
        encoded_src = encoded_src.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        
        pp = torch.cat([encoded_src[:, -1, :], pl], 1)
        # print(encoded_src.shape)
        # print("pl shape forward", pl.shape)
        # if pp.shape[1] == 128:
        #     print(pp)
        output = self.decoder(pp)  # Use only the last time step
        #dimension of the dec input last dim of encoder
        return output
    
    def sample_multiple(self, src, pl, num_samples=10):
        self.train()  # Enable dropout during inference
        samples = []
        for _ in range(num_samples):
            output = self.forward(src, pl)
            samples.append(output)
        self.eval()  # Optionally, set back to eval mode if needed elsewhere
        return torch.stack(samples)
        
    # def sample_autoregressive_multiple(self, src, pl, num_samples=10):
    #     # this is assuming we have pl shape that is 90, not 10 as in training.

    #     self.train()  # Enable dropout during inference
    #     samples = []

    #     for _ in range(num_samples):
    #         input_i = src
    #         outputs = []
            
    #         for i in range(9):
    #             pl_i = pl[:, i*10:(i+1)*10]
    #             output = self.forward(input_i, pl_i)
    #             output_reshaped = output.reshape([output.shape[0], 11, input_i.size(2)])[:, 1:,:] #only take new predictions, ignore first datapoint
    #             outputs.append(output_reshaped)
    #             input_i = torch.concat([input_i[:,10:,:], output_reshaped], axis=1)
    
    #         pred = torch.concat(outputs, axis=1)
    #         samples.append(pred)

    #     self.eval()  # Optionally, set back to eval mode if needed elsewhere
    #     #returns num_sample x batch size x forecast horizon x seq_length
    #     return torch.stack(samples)
    
    def sample_autoregressive_multiple(self, src, pl, num_samples=10):
        self.train()  # keep dropout on

        B, L, D = src.shape
        total_plen = pl.shape[1]
        horizon = 9 * (11 - 1)     # if each step adds 10 new points

        # 1) Repeat into a “samples” dimension
        src_rep = src.unsqueeze(0).expand(num_samples, B, L, D)
        pl_rep  = pl.unsqueeze(0).expand(num_samples, B, total_plen)

        # 2) Flatten samples & batch into one big batch
        #    new batch = num_samples * B
        big_batch = num_samples * B
        src_flat = src_rep.reshape(big_batch, L, D)
        pl_flat  = pl_rep.reshape(big_batch, total_plen)

        # 3) Single autoregressive loop over timesteps
        input_i = src_flat.to(self.device) 
        outputs = []
        for i in range(9):
            pl_i = pl_flat[:, i*10 : (i+1)*10].to(self.device)                # (big_batch, 10)
            out  = self.forward(input_i, pl_i)               # (big_batch, 11*D)
            # reshape: (big_batch, 11, D) → take the last 10 → (big_batch, 10, D)
            out = out.view(big_batch, 11, D)[:, 1:, :]

            outputs.append(out)
            # slide your window forward 10 steps
            input_i = torch.cat([input_i[:, 10:, :], out], dim=1)

        # concat all 9 blocks → (big_batch, horizon, D)
        preds_flat = torch.cat(outputs, dim=1)

        # 4) Un-flatten back to (num_samples, B, horizon, D)
        preds = preds_flat.view(num_samples, B, horizon, D)
        # print(f'generated {num_samples} trajectories')
        self.eval()
        return preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mbpo_uq")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="offline")
    # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--model_path" , type=str, default="saved_models")
    # parser.add_argument('-cuda', '--cuda_number', type=str, metavar='<device>', default=2, #required=True,
                        # help='Specify the CUDA device number to use.')
    parser.add_argument('-data_name', '--data_name', type=str, metavar='<size>', default='train',
                help='which data to work on.')
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=7,
                    help="Which GPU device index to use"
                )
    parser.add_argument("--iter", type=int, default=3)

    parser.add_argument("--task", type=str, default="Abiomed-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=50)


    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    #world transformer arguments
    parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=12,
                        help='Specify the sequence dimension.')
    parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=11*12,
                        help='Specify the sequence dimension.')
    parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=64,
                        help='Specify the batch size.') 
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=1, #change
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=2,
                help='Set the number of encoder layers.') 
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
                        help='Specify the learning rate.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
                help='Set the number of encoder layers.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='/data/abiomed_tmp/processed',
                        help='Specify the path to read data.')
    

    args = parser.parse_args()
    import random
    from torch.utils.tensorboard import SummaryWriter
    from common.logger import Logger
    from common.util import set_device_and_logger

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    # log_file = 'seed_1_0413_220409-Abiomed_v0_mbpo_uq_rerun'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

    model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)
    model_logger = Logger(writer=writer,log_path=model_path)

    Devid = args.devid if args.device == 'cuda' else -1
    set_device_and_logger(Devid, logger, model_logger)

    world_transformer = WorldTransformer(args, logger, pretrained = False)
    loader = world_transformer.read_data(mode='test')
    for src, pl, tgt in loader:
        preds = world_transformer.trained_model.sample_autoregressive_multiple(src, pl)
