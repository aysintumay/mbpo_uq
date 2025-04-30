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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from test import test
from train import train
from common.buffer import ReplayBuffer
from scoring import convert_tfenvents_to_csv, merge_csv
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
from common import util

import warnings
warnings.filterwarnings("ignore")

def main(args):
    
    run = wandb.init(
                project=args.task,
                group=args.algo_name,
                config=vars(args),
                )
    # run = None
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
    # log_file = 'seed_1_0413_220409-Abiomed_v0_mbpo_uq_rerun'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

    model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)
    model_logger = Logger(writer=writer,log_path=model_path)

    Devid = args.devid if args.device == 'cuda' else -1
    set_device_and_logger(Devid, logger, model_logger)

    results = []

    for i in np.arange(args.iter):
        start_time = time.time()
        print(f"====================Iteration {i+1}====================")
        if i == 0:
        
            with open(f'/data/abiomed_tmp/intermediate_data_uambpo/discounted_trainset.pkl', 'rb') as f:
                offline_buffer_train = pickle.load(f)
            with open('/data/abiomed_tmp/intermediate_data_uambpo/discounted_testset.pkl', 'rb') as f:
                offline_buffer_test = pickle.load(f)
        
            stds = np.array([1.2599670e+01, 4.6925778e+02, 5.8842087e+01, 1.5025043e+01,
        1.5730153e+01, 2.3981575e+01, 1.2024239e+01, 2.2280893e+01,
        1.7170943e+02, 1.7599674e+01, 1.9673981e-01, 1.4662008e+01,
        2.1159306e+00])
            
            means = np.array([7.3452431e+01, 3.9981541e+03, 2.8203378e+02, 3.9766106e+01,
        1.0223494e+01, 9.2290756e+01, 6.1786270e+01, 3.2400185e+01,
        6.0808063e+02, 8.4936722e+01, 6.1181599e-01, 6.5555145e+01,
        6.0715165e+00])
        
            norm_info = {'rwd_stds':stds, 'rwd_means':means, 'scaler': None}

        os.makedirs(model_logger.log_path, exist_ok=True)

        args.pretrained = True
        args.data_name = 'train'  

        # args.crps_scale = 1
    
        #train on offline dataset or replayed dataset
        norm_info, trainer = train(i, logger, run, model_logger, args, norm_info, offline_buffer_train if offline_buffer_train is not None else None, )
        
        
        #save the policy
        policy = trainer.algo.policy
        # trainer.algo.policy.load_state_dict(policy)
        
        # os.makedirs(model_path, exist_ok=True)
        # #save policy
        # torch.save(policy.state_dict(), os.path.join(model_path, f"policy_v_1_{args.task}_{args.crps_scale}_{i}.pth"))
        # policy.to(util.device)
        #save transition model
        trainer.algo.save_dynamics_model(f"dynamics_model_{i}")
        
        if i == args.iter-1:
            args.crps_scale = None

        trainer._eval_episodes = 49939
        args.data_name = 'train'

        args.mode = 'offline'
        args.pretrained = True

        print('pretrained', args.pretrained, '\nstarted testing')
        print('log path ' , log_path)
        dataset_train, _ = test(i, args, model_logger, norm_info, policy, trainer, offline_buffer_train if offline_buffer_train is not None else None, log_path)

        #save the dataset
        if not os.path.exists('/data/abiomed_tmp/intermediate_data_uambpo'):
            os.makedirs('/data/abiomed_tmp/intermediate_data_uambpo')
        with open(os.path.join('/data/abiomed_tmp/intermediate_data_uambpo',f'dataset_train_v_{args.crps_scale}_{i+1}.pkl'), 'wb') as f:
            pickle.dump(dataset_train, f)

        #get renewed test dataset of 20k
        trainer._eval_episodes = 28015
        args.data_name = 'test'
        args.mode = 'online'
        args.pretrained = True
        print( 'pretrained', args.pretrained, '\nstarted testing')

        dataset_test, eval_info = test(i, args,model_logger,norm_info, policy, trainer, offline_buffer_test if offline_buffer_test is not None else None, log_path)
        #save the dataset
        with open(os.path.join('/data/abiomed_tmp/intermediate_data_uambpo',f'dataset_test_v_{args.crps_scale}_{i+1}.pkl'), 'wb') as f:
            pickle.dump(dataset_test, f)

        offline_buffer_train = dataset_train
        offline_buffer_test = dataset_test

        norm_info['scaler'] = None #RECALCULATE REWARD SCALER

        mean_return = np.mean(eval_info["eval/episode_reward"])
        std_return = np.std(eval_info["eval/episode_reward"])
        mean_length = np.mean(eval_info["eval/episode_length"])
        std_length = np.std(eval_info["eval/episode_length"])

        if args.task == 'Abiomed-v0':
            mean_accuracy = np.mean(eval_info["eval/episode_accuracy"])
            std_accuracy = np.std(eval_info["eval/episode_accuracy"])
            mean_1_off_accuracy = np.mean(eval_info["eval/episode_1_off_accuracy"])
            std_1_off_accuracy = np.std(eval_info["eval/episode_1_off_accuracy"])
            mean_1_mse = np.mean(eval_info["eval/mse"])
            std_1_mse = np.std(eval_info["eval/mse"])

            print(f"Iteration {i} - Seed {args.seed} - Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
            print(f"Iteration {i} - Seed {args.seed} - 1-off Accuracy: {mean_1_off_accuracy:.2f} ± {std_1_off_accuracy:.2f}")
            print(f"Iteration {i} - Seed {args.seed} - MSE: {mean_1_mse:.2f} ± {std_1_mse:.2f}")
            wandb.log({
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "mean_1_off_accuracy": mean_1_off_accuracy,
                "std_1_off_accuracy": std_1_off_accuracy,
                "mean_1_mse": mean_1_mse,
                "std_1_mse": std_1_mse,
                "seed": args.seed
            })

            results.append({
                'seed': args.seed,
                'mean_return': mean_return,
                'std_return': std_return,
                'mean_length': mean_length,
                'std_length': std_length,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'mean_1_off_accuracy': mean_1_off_accuracy,
                'std_1_off_accuracy': std_1_off_accuracy,
                'mean_1_mse': mean_1_mse,
                'std_1_mse': std_1_mse,
                'iter': i
            })
            time_total = time.time() - start_time
        
        wandb.log({
            "mean_return": mean_return,
            "std_return": std_return,
            "seed": args.seed
        })        
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
    # parser.add_argument('-cuda', '--cuda_number', type=str, metavar='<device>', default=2, #required=True,
                        # help='Specify the CUDA device number to use.')
    parser.add_argument('-data_name', '--data_name', type=str, metavar='<size>', default='train',
                help='which data to work on.')
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=0,
                    help="Which GPU device index to use"
                )
    parser.add_argument("--iter", type=int, default=3)


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
    parser.add_argument("--reward-penalty-coef", type=float, default=1) #1e=6
    parser.add_argument("--rollout-length", type=int, default=3) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=5000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--crps_scale', type=float, default=None)
    # parser.add_argument('--iterations', type=int, default=4)


    parser.add_argument("--epoch", type=int, default=50) #1000 #change
    parser.add_argument("--step-per-epoch", type=int, default=1000) # will be equated to #of samples in train and test #change
    parser.add_argument("--eval_episodes", type=int, default=1000) # #change
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
