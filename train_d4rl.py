import argparse
import datetime
import pickle
import os
import random
import importlib
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
from common import util
from abiomed_env import AbiomedEnv



def train(i, logger, run, env, args, dataset = None, ):


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config
    

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        # target_entropy = args.target_entropy if args.target_entropy \
        #     else -np.prod(env.action_space.shape)
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )

    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     device = args.device,
                                     **config["transition_params"]
                                     )    
      

    # if i == 0:
      
    #     print('Policy to be trained!')
    # else:
    #     # load offline buffer
        
    #     policy_state_dict = torch.load(os.path.join(model_logger.log_path, f'policy_{args.task}_{i-1}.pth'))
    #     sac_policy.load_state_dict(policy_state_dict)
        
    #     sac_policy.to(args.device)
    #     print(f'Policy loaded from {model_logger.log_path}!')
    #     # dynamics_model.load_model(f'dynamics_model_{i-1}') 
      

    offline_buffer = ReplayBuffer(
            buffer_size=len(dataset["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32
        )

    offline_buffer.load_dataset(dataset)

    model_buffer = ReplayBuffer(
            buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32
        )
    # create MOPO algo
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        # rollout_batch_size=args.rollout_batch_size,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"]
    )
    #load world model

    # dynamics_model.load_model(f'dynamics_model') 

    
    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        run_id = run.id,
        env_name = args.task,
        eval_episodes=args.eval_episodes,
        terminal_counter= args.terminal_counter if args.task == "Abiomed-v0" else None,
        ite = i  
    )

    # pretrain dynamics model on the whole dataset
    trainer.train_dynamics()
    #  

    
    # begin train
    trainer.train_policy()

    
    return  trainer


if __name__ == "__main__":

    
    train()
