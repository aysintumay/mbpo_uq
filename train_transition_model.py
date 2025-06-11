#!/usr/bin/env python3
import argparse
import os
import pickle
import importlib

import gym
import d4rl
import numpy as np
import torch
import torch.nn as nn

from common.normalizer import StandardNormalizer
from common.buffer import ReplayBuffer
from common.functional import dict_batch_generator
from models.transition_model import TransitionModel

# If your world‐model variant is needed, uncomment:
# from models.d4rl_world_model import D4RLWorldModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn_dynamics(dynamics_model, offline_buffer, params, device):
    """
    Trains the dynamics_model on data from offline_buffer, using the hyperparameters in params.
    Returns the trained dynamics_model.
    """
    # Split buffer into train / eval by holdout_ratio
    max_sample_size = offline_buffer.get_size
    num_train_data = int(max_sample_size * (1.0 - params["holdout_ratio"]))
    env_data = offline_buffer.sample_all()

    train_data, eval_data = {}, {}
    for key in env_data.keys():
        train_data[key] = env_data[key][:num_train_data]
        eval_data[key] = env_data[key][num_train_data:]

    dynamics_model.reset_normalizers()
    dynamics_model.update_normalizer(
        train_data["observations"], train_data["actions"]
    )

    # Training loop
    model_tot_train_timesteps = 0
    model_train_iters = 0
    model_train_epochs = 0
    num_epochs_since_prev_best = 0
    break_training = False

    dynamics_model.reset_best_snapshots()

    # Initial eval
    print("Start training dynamics")
    eval_mse_losses, _ = dynamics_model.eval_data(
        eval_data, update_elite_models=False
    )
    print(
        f"Initial eval MSE: {eval_mse_losses.mean():.6f}  (timesteps=0)"
    )
    dynamics_model.update_best_snapshots(eval_mse_losses)

    while not break_training:
        # 1) Run through one epoch of batches
        for train_data_batch in dict_batch_generator(
            train_data, params["model_batch_size"]
        ):
            # Move tensors to correct device
            train_data_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in train_data_batch.items()
            }
            dynamics_model.update(train_data_batch)
            model_train_iters += 1
            model_tot_train_timesteps += 1

        # 2) Evaluate on holdout
        eval_mse_losses, _ = dynamics_model.eval_data(
            eval_data, update_elite_models=False
        )
        print(
            f" Eval MSE: {eval_mse_losses.mean():.6f}, after {model_tot_train_timesteps} timesteps,"
        )
        updated = dynamics_model.update_best_snapshots(eval_mse_losses)
        num_epochs_since_prev_best += 1

        if updated:
            model_train_epochs += num_epochs_since_prev_best
            num_epochs_since_prev_best = 0

        # 3) Early‐stop conditions
        if (
            num_epochs_since_prev_best
            >= params["max_model_update_epochs_to_improve"]
            or model_train_iters > params["max_model_train_iterations"]
            or model_tot_train_timesteps > 800000
        ):
            break

    # Load best ensemble snapshots & final eval
    dynamics_model.load_best_snapshots()
    dynamics_model.eval_data(eval_data, update_elite_models=True)

    # Log some normalizer stats (optional)
    model_log_infos = {}
    model_log_infos["misc/norm_obs_mean"] = torch.mean(
        torch.Tensor(dynamics_model.obs_normalizer.mean)
    ).item()
    model_log_infos["misc/norm_obs_var"] = torch.mean(
        torch.Tensor(dynamics_model.obs_normalizer.var)
    ).item()
    model_log_infos["misc/norm_act_mean"] = torch.mean(
        torch.Tensor(dynamics_model.act_normalizer.mean)
    ).item()
    model_log_infos["misc/norm_act_var"] = torch.mean(
        torch.Tensor(dynamics_model.act_normalizer.var)
    ).item()
    model_log_infos["misc/model_train_epochs"] = model_train_epochs
    model_log_infos["misc/model_train_train_steps"] = model_train_iters

    return dynamics_model, dynamics_model.obs_normalizer, dynamics_model.act_normalizer




def main(args):
    data_path = args.data_path
    noise_level = args.noise

    env_name = "hopper-expert-v0"
    env = gym.make(env_name)

    # 1) Load or fetch dataset
    if data_path == "":
        print("No data_path provided; loading from d4rl.qlearning_dataset...")
        raw_dataset = d4rl.qlearning_dataset(env)
        # Convert NumPy arrays to torch.Tensors
        dataset = {
            key: torch.tensor(raw_dataset[key], dtype=torch.float)
            for key in raw_dataset.keys()
        }
    else:
        print(f"Loading pickled dataset from '{data_path}' ...")
        with open(data_path, "rb") as f:
            raw = pickle.load(f)
        dataset = {key: torch.tensor(raw[key], dtype=torch.float) for key in raw.keys()}
    
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)

    # 2) Create a ReplayBuffer and load
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
    )
    offline_buffer.load_dataset(dataset)

    # 3) Build hyperparameters for the transition model
    transition_params = {
        "model_batch_size": 256,
        "use_weight_decay": True,
        "optimizer_class": "Adam",
        "learning_rate": 0.001,
        "holdout_ratio": 0.2,
        "inc_var_loss": True,
        "model": {
            "hidden_dims": [200, 200, 200, 200],
            "decay_weights": [
                0.000025,
                0.00005,
                0.000075,
                0.000075,
                0.0001,
            ],
            "act_fn": "swish",
            "out_act_fn": "identity",
            "num_elite": 5,
            "ensemble_size": 7,
        },
    }

    # If you need MOPO parameters, you can merge them here (but they are unused in this script)
    mopo_params = {
        "max_epoch": 125,
        "rollout_batch_size": 50000,
        "rollout_mini_batch_size": 10000,
        "model_retain_epochs": 1,
        "num_env_steps_per_epoch": 1000,
        "train_model_interval": 250,
        "max_trajectory_length": 1000,
        "eval_interval": 1000,
        "num_eval_trajectories": 10,
        "snapshot_interval": 2000,
        "model_env_ratio": 0.95,
        "max_model_update_epochs_to_improve": 5,
        "max_model_train_iterations": np.inf,
        "hold_out_ratio": 0.1,
    }
    params = {**transition_params, **mopo_params}

    # 4) Instantiate the transition model
    task = env_name.split("-")[0]  # e.g. "hopper"
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns

    transition_model = TransitionModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        static_fns=static_fns,
        lr=transition_params["learning_rate"],
        device=args.device,
        **transition_params,
    )

    # 5) Train
    trained_model, obs_normalizer, act_normalizer = learn_dynamics(transition_model, offline_buffer, params, args.device)
    if args.noise > 0:
        env_name =  env_name+"_noisy"
    # 6) Save under dynamics_model_{noise_level}.pt
    save_dir = os.path.join("saved_models", env_name)
    os.makedirs(save_dir, exist_ok=True)

    # Use noise level formatted without extra decimals, e.g. "0.1"
    noise_str = str(noise_level)
    
    save_path = os.path.join(save_dir, f"transition_model/dynamics_model_{noise_str}.pt")
    torch.save({
            'model_state_dict': trained_model.networks["model"].state_dict(),
            'obs_normalizer': obs_normalizer,
            'act_normalizer': act_normalizer
        }, save_path)
    print(f"Saved trained model to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a dynamics model on a D4RL dataset (e.g. Hopper)."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--noise",
        type=float,
        default = 0.0,
        help="Noise level (e.g. 0.1). Used for naming the saved model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help=
            "Path to a pickled dataset (e.g. " 
            "'hopper-expert-v0_noisy_0.1_unnorm.pkl'). "
            "If omitted, it will load from d4rl.qlearning_dataset(env)."
        ),
    
    args = parser.parse_args()

    noise_level = args.noise
    data_path = args.data_path
    main(args)
