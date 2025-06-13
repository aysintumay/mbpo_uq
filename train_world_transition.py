import argparse
import datetime
import os
import random
import importlib
import wandb 

import gym
import d4rl
import abiomed_env
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

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
from models.d4rl_transition_world_model import D4RLWorldModel



def train(run, seed, args):

    # create env and dataset
    scaler_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
    if args.task == "Abiomed-v0":
        gym.envs.registration.register(
        id='Abiomed-v0',
        entry_point='abiomed_env:AbiomedEnv',  
        max_episode_steps = 1000,
        )
        kwargs = {"args": args, "logger": logger, 'scaler_info': scaler_info}
        env = gym.make(args.task, **kwargs)
        dataset = env.qlearning_dataset()
    else:
        env = gym.make(args.task)
        dataset = d4rl.qlearning_dataset(env)
        env.id = ""
        print("eval id is", env.id)

    #CHANGE
    # dataset = {k: v[:5] for k, v in dataset.items()}

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    

    env.seed(seed)


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

    actor = ActorProb(actor_backbone, dist, util.device)
    critic1 = Critic(critic1_backbone, util.device)
    critic2 = Critic(critic2_backbone, util.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        # target_entropy = args.target_entropy if args.target_entropy \
        #     else -np.prod(env.action_space.shape)
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
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
        device=util.device
    )

    #we load the world model as the transition model here
    dynamics_model = D4RLWorldModel(args.task)
    dynamics_model.load_model(args.transition_model_path)

    # create buffer
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
        # logger=logger,
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
        # logger=logger,
        log_freq=args.log_freq,
        run_id = run.id if run!=None else 0,
        env_name = args.task,
        eval_episodes=args.eval_episodes,
        terminal_counter= args.terminal_counter if args.task == "Abiomed-v0" else None,
        
    )

    
    

    # policy_state_dict = torch.load(os.path.join(util.logger_model.log_path, f"policy_{args.task}.pth"))
    # sac_policy.load_state_dict(policy_state_dict)
    # begin train
    trainer.train_policy()

    if args.task == "Abiomed-v0":
        return  {
            'rwd_stds': env.rwd_stds,
            'rwd_means': env.rwd_means, 
            'scaler': env.scaler
            } , sac_policy, trainer, 
    else:
        return sac_policy, trainer

if __name__ == "__main__":
    args = get_args()
    

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

    Devid = args.devid if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger)

    run = None # no wandb for baselines

    train(run, logger, args)
