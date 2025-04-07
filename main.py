import argparse
import time
import os
import datetime
import random
import wandb
import numpy as np
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from test import test
from train import train, get_args
from common.buffer import ReplayBuffer
from scoring import convert_tfenvents_to_csv, merge_csv
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger


def main(args):

    run = wandb.init(
                project=args.task,
                group=args.algo_name,
                config=vars(args),
                )
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    Devid = 0 if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger)

    iterations = 3

    for i in range(iterations):

        if i == 0:
            args.pretrained = False
            # train_args.replay = False
        else:
            args.pretrained = True
            # train_args.replay = True


        #train on offline dataset or replayed dataset
        train(logger, args, offline_buffer_train if offline_buffer_train is not None else None, )

        #get renewed train dataset of 50k
        args.eval_episodes = 50000
        args.data_name = 'train'
        args.policy_path = args.policy_path

        args.mode = 'offline'
        dataset_train = test(i, args, offline_buffer_train if offline_buffer_train is not None else None, log_path)
        get_returns()

        #get renewed test dataset of 20k
        args.data_name = 'test'
        args.eval_episodes = 20000
        args.mode = 'online'

        dataset_test = test(i, args, offline_buffer_test if offline_buffer_test is not None else None)
        get_returns()

        obs_shape = 1080
        action_dim = 1
        offline_buffer_train = ReplayBuffer(
            buffer_size=len(dataset_train["observations"]),
            obs_shape=obs_shape,
            obs_dtype=np.float32,
            action_dim=action_dim,
            action_dtype=np.float32
        )

        offline_buffer_train.load_dataset(dataset_train)

        offline_buffer_test = ReplayBuffer(
            buffer_size=len(dataset_test["observations"]),
            obs_shape=obs_shape,
            obs_dtype=np.float32,
            action_dim=action_dim,
            action_dtype=np.float32
        )
        offline_buffer_test.load_dataset(dataset_test)

        def get_returns():

            """ 
            at the end of each test run, tfevents is generated
            convert tfevents to csv
            seed_a>ite_i>offline or online> all docs
            generate csv for each ite
            generate merged csv for all iterations
            """

            path = os.path.join(log_path, 'test',  f'ite_{i}', args.mode)
            result = convert_tfenvents_to_csv(path, args.xlabel, args.ylabel, args.ylabel2 )
            merge_csv(result, path, args.xlabel, args.ylabel, args.ylabel2)
        
        


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mopo")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="offline")
    # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--policy_path" , type=str, default="log/halfcheetah_medium_plot/mopo/seed_5_0403_230811-halfcheetah_medium_replay_v0_mopo/policy_halfcheetah-medium-replay-v0.pth")

    parser.add_argument("--task", type=str, default="Abiomed-v0")
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
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0) #1e=6
    parser.add_argument("--rollout-length", type=int, default=5) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=50000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=1000) #1000
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=20,
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=2,
                help='Set the number of encoder layers.') 
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
                        help='Specify the learning rate.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0,
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
    parser.add_argument(
        '--task', default='halfcheetah-medium-v2', help='task'
    )
    parser.add_argument(
        '--algos', default="mopo", help='algos'
    )
    
    parser.add_argument(
        '--xlabel', default='Timesteps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='episode_reward', help='matplotlib figure ylabel'
    )

    parser.add_argument(
        '--ylabel2', default='normalized_episode_reward', help='matplotlib figure ylabel'
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

  
    main(args=get_args())
