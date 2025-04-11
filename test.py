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
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer,plot_accuracy
from common.util import set_device_and_logger
# from trainer import _evaluate, evaluate




def _evaluate(policy, eval_env, episodes):
        policy.eval()
        obs = eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < episodes:
            action = policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )

                #d4rl don't have REF_MIN_SCORE and REF_MAX_SCORE for v2 environments
                dset_name = eval_env.unwrapped.spec.name+'-v0'
                print(d4rl.get_normalized_score(dset_name, np.array(episode_reward))*100)

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = eval_env.reset()

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }






def get_eval(policy, env, logger, trainer, args,):

    reward_l, acc_l, off_acc = [], [], []
    reward_std_l, acc_std_l, off_acc_std = [], [], []

    trainer.eval_env = env
    trainer.algo.policy = policy

    if args.task == 'Abiomed-v0':
        eval_info, dataset = trainer.evaluate()
    else:

        #TODO: add eval function for d4rl   
        eval_info, dataset = _evaluate(policy, env, args.eval_episodes)


    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    if args.task == 'Abiomed-v0':
        ep_accuracy_mean, ep_accuracy_std = np.mean(eval_info["eval/episode_accuracy"]), np.std(eval_info["eval/episode_accuracy"])
        ep_1_off_accuracy_mean, ep_1_off_accuracy_std = np.mean(eval_info["eval/episode_1_off_accuracy"]), np.std(eval_info["eval/episode_1_off_accuracy"])

    if args.task == 'Abiomed-v0':
        logger.record("eval/episode_accuracy", ep_accuracy_mean, args.eval_episodes, printed=False)
        # logger.record("eval/episode_1_off_accuracy", ep_1_off_accuracy_mean, args.eval_episodes, printed=False)
        logger.print(f"episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f},\
                            episode_accuracy: {ep_accuracy_mean:.3f} ± {ep_accuracy_std:.3f},\
                            episode_1_off_accuracy: {ep_1_off_accuracy_mean:.3f} ± {ep_1_off_accuracy_std:.3f}")
    else:
        logger.print(f"episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
    logger.record("eval/episode_reward", ep_reward_mean, args.eval_episodes, printed=False)
    # logger.record("eval/episode_length", ep_length_mean, args.eval_episodes, printed=False)
    # if env.name == 'Abiomed-v0':
    #     plot_accuracy(np.array(reward_l), np.array(reward_std_l)/args.eval_episodes, 'Average Return')
    #     plot_accuracy(np.array(acc_l), np.array(acc_std_l)/args.eval_episodes, 'Accuracy')
    #     plot_accuracy(np.array(off_acc), np.array(off_acc_std)/args.eval_episodes, '1-off Accuracy')

    if args.task != 'Abiomed-v0':
        dset_name = env.unwrapped.spec.name+'-v0'
        normalized_score_mean = d4rl.get_normalized_score(dset_name, ep_reward_mean)*100
        normalized_score_std = d4rl.get_normalized_score(dset_name, ep_reward_std)*100
        logger.record("normalized_episode_reward", normalized_score_mean, ep_length_mean, printed=False)
        logger.print(f"normalized_episode_reward: {normalized_score_mean:.3f} ± {normalized_score_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
    # else:
        # logger.record("episode_reward", ep_reward_mean/ep_length_mean, args.eval_episodes, printed=False)

    # if args.task == 'Abiomed-v0':
    #     logger.record("avg episode_accuracy", np.array(acc_l).mean(), args.eval_episodes, printed=False)
    #     logger.record("avg episode_1_off_accuracy", np.array(off_acc).mean(), args.eval_episodes, printed=False)
    return dataset



def test(i, run, args, model_logger, norm_info, offline_buffer=None, log_path=None):


    log_path = os.path.join(log_path, 'test',  f'ite_{i}', args.mode)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    Devid = 0 if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger, model_logger)

    # create env and dataset
   
    if args.task == "Abiomed-v0":
        # Register the environment only once
        gym.envs.registration.register(
            id='Abiomed-v0',
            entry_point='abiomed_env:AbiomedEnv',  
            max_episode_steps=1000,
        )
        # Build kwargs based on whether offline_buffer is provided
        kwargs = {"args": args, "logger": logger, "data_name": "test", 'scaler_info': norm_info}
        if offline_buffer is not None:
            kwargs["offline_buffer"] = offline_buffer
        env = gym.make(args.task, **kwargs)
    else:
        env = gym.make(args.task)

    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    
    env.seed(args.seed)


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
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

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
    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     **config["transition_params"]
                                     )    
    
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"]
    )
   
    # create trainer
    trainer = Trainer(
        algo,
        # world_model,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        run_id = run.id,
        eval_episodes=args.eval_episodes,
        
    )
    
    policy_state_dict = torch.load(os.path.join(model_logger.log_path, f'policy_{args.task}.pth'))
    sac_policy.load_state_dict(policy_state_dict)

    get_eval(sac_policy, env, logger, trainer, args)

    return dataset

if __name__ == "__main__":

    
    test()
