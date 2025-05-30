import argparse
import pickle
import time
import os
import datetime
import random
import wandb
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import gym
import d4rl

from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer
from test_d4rl import test
from train_d4rl import train
from common.logger import Logger
from common.util import set_device_and_logger
from models.d4rl_world_model import D4RLWorldModel

import warnings
warnings.filterwarnings("ignore")



def bundle_buffers(orig_data, dataset_uambpo, args, real_ratio=0.8):
    buffer_uambpo = ReplayBuffer(
            buffer_size=len(dataset_uambpo["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32
        )

    buffer_uambpo.load_dataset(dataset_uambpo)

    buffer_orig = ReplayBuffer(
            buffer_size=len(orig_data["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32
        )

    buffer_orig.load_dataset(orig_data)

    real_sample_size = int(len(orig_data['observations']) * real_ratio)
    fake_sample_size = len(dataset_uambpo['observations']) - real_sample_size
    real_batch = buffer_orig.sample(batch_size=real_sample_size)
    fake_batch = buffer_uambpo.sample(batch_size=fake_sample_size)
    data = {
    'observations': np.concatenate([real_batch['observations'], fake_batch['observations']], axis=0),
    'actions': np.concatenate([real_batch['actions'], fake_batch['actions']], axis=0),
    'next_observations':np.concatenate([real_batch['next_observations'], fake_batch['next_observations']], axis=0),
    'rewards':np.concatenate([real_batch['rewards'], fake_batch['rewards']], axis=0),
    'terminals': np.concatenate([real_batch['terminals'], fake_batch['terminals']], axis=0)
    }
    return data

def initialize_env_and_seeds(args):
    env = gym.make(args.task)
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return env

def setup_loggers(args, t0):
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    logger = Logger(writer=writer, log_path=log_path)
    model_logger = Logger(writer=writer, log_path=model_path)
    return writer, logger, model_logger, log_path

def print_results(eval_info, results, args, i):
    mean_return = np.mean(eval_info["eval/episode_reward"])
    std_return = np.std(eval_info["eval/episode_reward"])
    mean_length = np.mean(eval_info["eval/episode_length"])
    std_length = np.std(eval_info["eval/episode_length"])

    results.append({
        'seed': args.seed,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
        'std_length': std_length,
        'iter': i
    })
    
    wandb.log({
        "mean_return": mean_return,
        "std_return": std_return,
        "seed": args.seed
        })    
    return mean_return, std_return, results 

def main(args):
    
    run = wandb.init(
                project=args.task,
                group=args.algo_name,
                config=vars(args),
                )

    env = initialize_env_and_seeds(args)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)    
    
    
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")

    writer, logger, model_logger, log_path = setup_loggers(args, t0)

    Devid = args.devid if args.device == 'cuda' else -1
    device_model = set_device_and_logger(Devid, logger, model_logger)
    args.device = device_model
    world_model = D4RLWorldModel(args.task, device = args.device)
    world_model.load_model(args.world_model_path)

    results = []
    with open(f'/data/abiomed_tmp/intermediate_data_d4rl/{args.task}_crps_discount_{args.discount_factor}.pkl', 'rb') as f:
                offline_buffer_train = pickle.load(f)

    for i in np.arange(args.iter):
        start_time = time.time()
        print(f"====================Iteration {i+1}====================")
        
        # offline_buffer_train = {key: offline_buffer_train[key][:1000] for key in offline_buffer_train.keys()}

        os.makedirs(model_logger.log_path, exist_ok=True)

        # args.pretrained = True
    
        trainer = train(i, logger, run, env, args, offline_buffer_train if offline_buffer_train is not None else None, )
        
        #save the policy
        policy = trainer.algo.policy
        trainer.algo.save_dynamics_model(f"dynamics_model_{i}")
        
        if i == args.iter-1:
            args.crps_scale = 0

        # trainer._eval_episodes = len(offline_buffer_train['observations'])
        args.data_name = 'train'

        args.mode = 'offline'
        args.pretrained = True

        print('pretrained', args.pretrained, '\nstarted testing')
        print('log path ' , log_path)
        dataset_train, eval_info = test(i,args, world_model, model_logger, policy, trainer, offline_buffer_train if offline_buffer_train is not None else None, log_path)

        #save the dataset
        # if not os.path.exists('/data/abiomed_tmp/intermediate_data_d4rl'):
        #     os.makedirs('/data/abiomed_tmp/intermediate_data_d4rl')
        # with open(os.path.join('/data/abiomed_tmp/intermediate_data_d4rl',f'{args.task}_{args.crps_scale}_{i+1}.pkl'), 'wb') as f:
        #     pickle.dump(dataset_train, f)

        offline_buffer_train = bundle_buffers(offline_buffer_train, dataset_train, args, real_ratio=args.real_ratio)
        
        # offline_buffer_train = dataset_train
        mean_return, std_return, results = print_results(eval_info, results, args, i)
        time_total = time.time() - start_time
        print(f"Iteration {i} - Seed {args.seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        print('Iteration', i, 'time:', time_total)

    os.makedirs(os.path.join('results', args.task, 'uambpo'), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, 'uambpo', f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
   
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mbpo_uq")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="offline")
    # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--policy_path" , type=str, default="")
    parser.add_argument("--model_path" , type=str, default="saved_models")
    parser.add_argument('--world_model_path',type=str, default="saved_models/halfcheetah-random-v0/world_model_0.55.pth")

    parser.add_argument('-data_name', '--data_name', type=str, metavar='<size>', default='train',
                help='which data to work on.')
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=4,
                    help="Which GPU device index to use"
                )
    parser.add_argument("--iter", type=int, default=3)


    parser.add_argument("--task", type=str, default="halfcheetah-random-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-3) #-action_dim
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=0) #1e=6
    parser.add_argument("--rollout-length", type=int, default=3) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=50000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--crps_scale', type=float, default=0.05)
    parser.add_argument('--discount_factor', type=float, default=0.1)
    parser.add_argument('--real_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size_generation', type=float, default=50)
    # parser.add_argument('--iterations', type=int, default=4)


    parser.add_argument("--epoch", type=int, default=100) #1000 #change
    parser.add_argument("--step-per-epoch", type=int, default=1000) # will be equated to #of samples in train and test #change
    parser.add_argument("--eval_episodes", type=int, default=10) # #change
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--terminal_counter", type=int, default=1) 

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
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=20, #change
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
    
    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )
   
    args = parser.parse_args()

    return args


if __name__ == "__main__":

  
    main(args=get_args())
