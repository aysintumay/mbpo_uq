# Borrow a lot from tianshou:
# https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/plotter.py
import csv
import os
import re
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tqdm
import argparse

from tensorboard.backend.event_processing import event_accumulator

from abiomed_env import AbiomedEnv

COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)


def convert_tfenvents_to_csv(root_dir, xlabel, ylabel):
    """Recursively convert test/metric from all tfevent file under root_dir to csv."""
    tfevent_files = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(re.compile(r"^.*tfevents.*$"), absolute_path):
                tfevent_files.append(absolute_path)
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(os.path.split(tfevent_file)[0], ylabel+'.csv')
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            content = [[xlabel, ylabel]]
            for test_rew in ea.scalars.Items('eval/'+ylabel):
                content.append(
                    [
                        round(test_rew.step, 4),
                        round(test_rew.value, 4),
                    ]
                )
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, xlabel, ylabel):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        [xlabel, ylabel+'_mean', ylabel+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, ylabel+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)


def csv2numpy(file_path):
    df = pd.read_csv(file_path)
    step = df.iloc[:,0].to_numpy()
    mean = df.iloc[:,1].to_numpy()
    std = df.iloc[:,2].to_numpy()
    return step, mean, std


def smooth(y, radius=0):
    convkernel = np.ones(2 * radius + 1)
    out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
    return out


def plot_figure(root_dir, task, algo_list, x_label, y_label, title, smooth_radius, color_list=None):
    fig, ax = plt.subplots()
    if color_list is None:
        color_list = [COLORS[i] for i in range(len(algo_list))]
    for i, algo in enumerate(algo_list):
        x, y, shaded = csv2numpy(os.path.join(root_dir, task, algo, y_label+'.csv'))
        # y = smooth(y, smooth_radius)
        # shaded = smooth(shaded, smooth_radius)
        # x=smooth(x, smooth_radius)
        ax.plot(x, y, color=color_list[i], label=algo_list[i])
        ax.fill_between(x, y-shaded, y+shaded, color=color_list[i], alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    

def plot_histogram(data, y_label,):
    color = [COLORS[-2],
             COLORS[-3],
             COLORS[-5], 
             COLORS[-6],
               COLORS[-7], 
               COLORS[-8]
               ]
    """
    Plot histograms of data[1][y_label], data[2][y_label], data[3][y_label],
    each in its own color.
    
    Parameters
    ----------
    data : dict of DataFrame-like
        data[i][y_label] should be iterable of values.
    y_label : str
        Column/key to plot.
    """
    plt.figure(figsize=(8, 5))
    
    # Loop over the three iterations
    labels = ['D_env', 'D_1'] + [f'Iteration {i}' for i in range(1, len(data))]

    for (i, c, lbl) in zip(range(len(data)), color, labels):
        plt.hist(
            np.round(data[i][y_label]),
            bins=50,
            density=True,
            alpha=0.6,
            color=c,
            label=lbl,
        )

    
    plt.xlabel(y_label)
    plt.ylabel('Count')
    plt.title(f'Histogram of {y_label} over {i} iterations')
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(args.output_path, f'{y_label}.png')
    plt.savefig(out_file, dpi=args.dpi, bbox_inches='tight')
    plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )
    parser.add_argument(
        '--task', default='Abiomed-v0', help='task'
    )
    parser.add_argument(
        '--algos', default=["mopo"], help='algos'
    )
    parser.add_argument(
        '--title', default=None, help='matplotlib figure title (default: None)'
    )
    parser.add_argument(
        '--xlabel', default='Timesteps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='episode_reward', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--hist_ylabel', default='actions', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--hist_ylabel2', default='rewards', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--smooth', type=int, default=10, help='smooth radius of y axis (default: 0)'
    )
    parser.add_argument(
        '--colors', default=None, help='colors for different algorithms'
    )
    parser.add_argument('--show', action='store_true', help='show figure')
    parser.add_argument(
        '--output-path', type=str, help='figure save path', default="results"
    )
    parser.add_argument(
        '--data-path', type=str, help='figure save path', default="/data/abiomed_tmp/intermediate_data_uambpo"
    )

    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--devid', type=int, default=0, help='device id')
    parser.add_argument('--algo_name', type=str, default='mopo', help='algorithm name')
    parser.add_argument('--logdir', type=str, default='log', help='log dir')
    parser.add_argument('--model_path', type=str, default='saved_models', help='model path')
    parser.add_argument(
        '--dpi', type=int, default=200, help='figure dpi (default: 200)'
    )
    args = parser.parse_args()

    # for algo in args.algos:
    #     path = os.path.join(args.root_dir, args.task, algo)
    #     result = convert_tfenvents_to_csv(path, args.xlabel, args.ylabel)
    #     merge_csv(result, path, args.xlabel, args.ylabel)

    # # plt.style.use('seaborn')
    # plot_figure(root_dir=args.root_dir, task=args.task, algo_list=args.algos, x_label=args.xlabel, y_label=args.ylabel, title=args.title, smooth_radius=args.smooth, color_list=args.colors)
    # if args.output_path:
    #     plt.savefig(os.path.join(args.output_path, 'return.png'), dpi=args.dpi, bbox_inches='tight')
    # if args.show:
    #     plt.show()

    #plot p-lvl and rewards by opening the pkl's file
    std_act = 2.1159306e+00
    mean_act = 6.0715165e+00
    import datetime
    from torch.utils.tensorboard import SummaryWriter
    from common.logger import Logger
    from common.util import set_device_and_logger
    import gym

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

    norm_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
    args.crps_scale = None
    args.data_name = 'train'
    # Register the environment only once
    gym.envs.registration.register(
        id='Abiomed-v0',
        entry_point='abiomed_env:AbiomedEnv',  
        max_episode_steps=1000,
    )
    # Build kwargs based on whether offline_buffer is provided
    kwargs = {"args": args, "logger": logger, "scaler_info": norm_info}
    env = gym.make(args.task, **kwargs)

    args.data_name = 'test'
    norm_info = {'rwd_stds': env.rwd_stds, 'rwd_means': env.rwd_means, 'scaler':  env.scaler}
    kwargs = {"args": args, "logger": logger, "scaler_info": norm_info}
    env = gym.make(args.task, **kwargs)
    dataset = env.data

    data_paths = [
                #   os.path.join(args.data_path, f"dataset_test_0.pkl"), 
                  os.path.join(args.data_path, f"discounted_testset.pkl"), #10k
                  os.path.join(args.data_path, f"dataset_test_v_1.pkl"), 
                    os.path.join(args.data_path, f"dataset_test_v_2.pkl"),
                    os.path.join(args.data_path, f"dataset_test_v_3.pkl"),
                    os.path.join(args.data_path, f"dataset_test_v_4.pkl"),
                  
                ]

    data = {}
    data[0] = env.data
    # Normalize the actions 
    data[0]['actions'] = data[0]['actions']*env.rwd_stds[12] + env.rwd_means[12] 
    data[0]['rewards'] = env.scaler.inverse_transform(data[0]['rewards'])
    i = 1
    for path in data_paths:
        
        with open(path, 'rb') as f:
            data[i] = pickle.load(f)
            # if i == 0:
                
        i += 1

    data[0]['rewards'] = data[0]['rewards'][:5000]
    data[0]['actions'] = data[0]['actions'][:5000]
    data[1]['rewards'] = data[1]['rewards'][:5000]
    data[1]['actions'] = data[1]['actions'][:5000]

    # with open(data_paths[0], 'rb') as f:
    #     data[0] = pickle.load(f)
    plot_histogram(data, args.hist_ylabel)
    
    plot_histogram(data, args.hist_ylabel2)
        
