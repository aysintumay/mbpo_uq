import argparse
import datetime
import pickle
import os
import random
import importlib
import tqdm
import time
# import wandb 

import gym
import d4rl
import abiomed_env
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
from common import util
from abiomed_env import AbiomedEnv
import random
from torch.utils.tensorboard import SummaryWriter
from common.logger import Logger
from common.util import set_device_and_logger




def main(i, logger, args, scaler_info,eval_episodes, offline_buffer = None, ):


# create env and dataset
    if args.task == "Abiomed-v0":
        # Register the environment only once
        gym.envs.registration.register(
            id='Abiomed-v0',
            entry_point='abiomed_env:AbiomedEnv',  
            max_episode_steps=1000,
        )
        # Build kwargs based on whether offline_buffer is provided
        kwargs = {"args": args, "logger": logger, "scaler_info": scaler_info}
        if offline_buffer is not None:
            kwargs["offline_buffer"] = offline_buffer
        env = gym.make(args.task, **kwargs)
        # dataset = env.qlearning_dataset()
        dataset = env.data
       
    else:
        env = gym.make(args.task)

        dataset = d4rl.qlearning_dataset(env)
    eval_info, dataset = generate_data(dataset, env, eval_episodes)
    # Save the dataset to a pickle file
    with open(f"/data/abiomed_tmp/intermediate_data_uambpo/discounted_{args.data_name}set.pkl", "wb") as f:
        pickle.dump(dataset, f)
    #save the eval_info
    with open(f"/data/abiomed_tmp/intermediate_data_uambpo/eval_info_{args.data_name}set.pkl", "wb") as f:
        pickle.dump(eval_info, f)
    return {'rwd_stds':env.rwd_stds, 'rwd_means':env.rwd_means, 'scaler':env.scaler}

def generate_data(dataset, env, eval_episodes):

    model = env.world_model.trained_model
    obs = env.reset().reshape(1,-1)
    
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0
    crps_list = []
    obs_ = []
    next_obs_ = []
    action_ = []
    full_action_ = []
    reward_ = []
    terminal_ = []
    N = dataset['observations'].shape[0]

    indx = np.random.choice(N,
                            size=eval_episodes,
                            replace=False)
    start_time = time.time()  
    for i in tqdm.tqdm(indx):
                         
        next_state_gt = env.get_next_obs()
        action = env.get_pl()
        action = action.repeat(90) #repeat the action for 90 steps
        full_pl = env.get_full_pl()
        next_obs, reward, terminal, info = env.step_crps(action) #next state predictions

        #MSE of next_obs and next_state_gt
        mse = ((next_obs.reshape(1,-1) - next_state_gt)**2).mean()*env.rwd_stds[12]

        #obs: (0,90) next_state_gt:(90,180) next_obs: (90,180), action: (90,180) act: (90,180)
        episode_reward += reward
        episode_length += 1       

       
    
        eval_ep_info_buffer.append(
            {"episode_reward": episode_reward,
                "episode_length": episode_length,
                # "episode_accurcy": acc, 
                # "episode_1_off_accuracy": acc_1_off,
                'mse': mse,
                "crps": info['crps'],
                }
        )
        num_episodes +=1
        # terminal_counter = 0
        episode_reward, episode_length = 0, 0
        # obs = self.eval_env.reset()
        # print("episode_reward", episode_reward, 
        #   "episode_length", episode_length,
        #   "episode_accuracy", acc_total/self._step_per_epoch, 
        #   "episode_1_off_accuracy", acc_1_off_total/self._step_per_epoch)
        # self.logger.print("EVAL TIME: {:.3f}s".format(time.time() - start_time))
        #obs, next_obs, reward, done

        obs_.append(list(obs[0]))
        next_obs_.append(list(next_state_gt.reshape(obs.shape)[0]))
        action_.append(action[0])
        full_action_.append(full_pl)
        reward_.append(reward)
        terminal_.append(terminal)

        if num_episodes != env.data['observations'].shape[0]:
            obs = env.get_obs().reshape(1,-1)
        else:
            break
    print(f'time spent  {time.time()-start_time}')
    #need actions to be unnormalized for plotting
    action_ = env.unnormalize(np.array(action_), idx=12)
    full_action_ = env.unnormalize(np.array(full_action_), idx=12).reshape(-1,90)
    dataset = {
            'observations': np.array(obs_),
            'actions': np.array(action_).reshape(-1, 1),  # Reshape to ensure it's 2D
            'rewards': np.array(reward_),
            'terminals': np.array(terminal_),
            'next_observations': np.array(next_obs_),
            'full_actions': np.array(full_action_), 
        }
    return {
        "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
        "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
        "eval/mse": [ep_info["mse"] for ep_info in eval_ep_info_buffer],
        "eval/crps": [ep_info["crps"] for ep_info in eval_ep_info_buffer],
    }, dataset


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
                    default=3,
                    help="Which GPU device index to use"
                )
    parser.add_argument("--iter", type=int, default=3)

    parser.add_argument("--task", type=str, default="Abiomed-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--crps_scale', type=float, default=0.3)


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
    

    args = parser.parse_args()
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # with open(f"/data/abiomed_tmp/intermediate_data_uambpo/discounted_trainset.pkl", "rb") as f:
    #     dataset = pickle.load(f)
    # #save the eval_info
    # with open(f"/data/abiomed_tmp/intermediate_data_uambpo/eval_info_trainset.pkl", "rb") as f:
    #     eval_info =pickle.load(f)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

    model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)
    model_logger = Logger(writer=writer,log_path=model_path)

    Devid = args.devid if args.device == 'cuda' else -1
    set_device_and_logger(Devid, logger, model_logger)

    scaler_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
    args.data_name = 'train'
    eval_episodes = 49939

    print('-----started generating train------')
    scaler_info = main(0, logger, args, scaler_info,eval_episodes, offline_buffer = None, ) 

    args.pretrained = True
    args.data_name = 'test'
    eval_episodes = 28015
    print('-----started generating test------')
    _ = main(0, logger, args, scaler_info, eval_episodes,offline_buffer = None, ) 
