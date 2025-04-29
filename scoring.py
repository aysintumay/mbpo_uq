import argparse
import csv
import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tqdm
import argparse

from tensorboard.backend.event_processing import event_accumulator


def convert_tfenvents_to_csv(root_dir, xlabel, ylabel, ylabel2):
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
            content = [[xlabel, ylabel, ylabel2]]
            for test_rew in ea.scalars.Items('eval/'+ylabel):
                content.append(
                    [
                        round(test_rew.step, 4),
                        round(test_rew.value, 4),
                    ]
                )
            # csv.writer(open(output_file, 'w')).writerows(content)
            

            for test_rew in ea.scalars.Items('eval/'+ylabel2):
                content[1].extend([round(test_rew.value, 4)])
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, xlabel, ylabel, ylabel2):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        [xlabel, ylabel+'_mean', ylabel+'_std', ylabel+'_2_mean', ylabel+'_2_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4), 
               round(array[:, 2].mean(), 4), round(array[:, 2].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, ylabel+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)

def unnorm_all(ts, rwd_stds, rwd_means):
    return ts  * rwd_stds +  rwd_means

def crps_evaluation(samples, y_true):

    """
    Calculate pointwise CRPS
    """
    e_x_y = np.mean(np.abs(samples - y_true), axis=0)
    e_x_x_prime = np.array([np.abs(si - sj) for i, si in enumerate(samples) for j, sj in enumerate(samples) if i != j]).mean(axis = 0)
    crps = e_x_y - 0.5 * e_x_x_prime
    crps_ed = crps

    # DO NOT, if calculating pointwise and concatenating with the feature vector 
    mean_crps = np.mean(crps)

    # mean_ed_2 = np.mean(samples, axis = 0)
    # std_ed_2 = np.std(samples, axis =0)

    # crps_ed = ps.crps_gaussian(samples, 
    #                        mean_ed_2,  
    #                        std_ed_2).mean()
    return mean_crps

def get_returns(log_path, i):

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


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
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
        '--ylabel2', default='episode_accuracy', help='matplotlib figure ylabel'
    )

    parser.add_argument("--log_path" , type=str, default="")


    args = parser.parse_args()


    for i in range(3):
        get_returns(args.log_path, i)