import time
import os
import wandb
import numpy as np
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm

from common import util


def plot_accuracy(mean_acc, std_acc, name=''):
    epochs = np.arange(mean_acc.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    ax.plot(epochs, mean_acc, label=f'{name }')
    ax.fill_between(epochs, mean_acc - std_acc/2, mean_acc + std_acc/2, alpha=0.5, label='± 1/2 Std')
    ax.set_xlabel('time')
    ax.set_ylabel(f'{name}')
    ax.set_title(f'{name} Over Epochs')
    ax.legend()
    wandb.log({f"{name}": wandb.Image(fig)})


def plot_p_loss(critic1,name=''):

    epochs = np.arange(critic1.shape[0])

    mean_c1 = critic1.mean(axis=1)
    std_c1 = critic1.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    ax.plot(epochs, mean_c1, label=f'{name} Loss')
    ax.fill_between(epochs, mean_c1 - std_c1/2, mean_c1 + std_c1/2, alpha=0.3, label='± 1/2 Std')
    ax.set_xlabel('time')
    ax.set_ylabel('Loss')
    ax.set_title(f'{name} Loss Over Time')
    ax.legend()
    wandb.log({f"{name} Loss": wandb.Image(fig)})


def plot_q_value(q1, name=''):


    epochs = np.arange(q1.shape[0])

    mean_c1 = q1.mean(axis=1)
    std_c1 = q1.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    ax.plot(epochs, mean_c1, label=f'{name} Value')
    ax.fill_between(epochs, mean_c1 - std_c1/2, mean_c1 + std_c1/2, alpha=0.3, label='± 1/2 Std')
    ax.set_xlabel('time')
    ax.set_ylabel('Loss')
    ax.set_title(f'{name} Value Over Time')
    ax.legend()
    wandb.log({f"{name} Value": wandb.Image(fig)})


class Trainer:
    def __init__(
        self,
        algo,
        # world_model,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        run_id,
        env_name = '',
        eval_episodes=10,
        terminal_counter=1,
        ite = 0
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self.run_id = run_id

        self.env_name = env_name
        # self.world_model = world_model 

        self._eval_episodes = eval_episodes
        self.terminal_counter = terminal_counter

        self.iter = ite

        run = wandb.init(project="abiomed",
                id=self.run_id,
                resume="allow"
                )
    def train_dynamics(self):
        start_time = time.time()
        self.algo.learn_dynamics()
        #self.algo.save_dynamics_model(
            #save_path=os.path.join(self.logger.writer.get_logdir(), "dynamics_model")
        #)
        self.algo.save_dynamics_model(f"dynamics_model")
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

    def train_policy(self):
        start_time = time.time()
        num_timesteps = 0
        # train loop
        q1_l, q2_l,q_l = [], [], []
        critic_loss1,critic_loss2,  actor_loss, entropy,alpha_loss = [], [],[], [], []
        reward_l, acc_l, off_acc = [], [], []
        reward_std_l, acc_std_l, off_acc_std = [], [], []
        for e in range(1, self._epoch + 1):
            self.algo.policy.train()
            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    if num_timesteps % self._rollout_freq == 0:
                        self.algo.rollout_transitions()
                    # update policy by sac
                    loss,q_values = self.algo.learn_policy()
                    q1_l.append(q_values['q1'])
                    q2_l.append(q_values['q2'])
                    q_l.append(q_values['q_target'])
                    critic_loss1.append(loss["loss/critic1"])
                    critic_loss2.append(loss["loss/critic2"])
                    actor_loss.append(loss["loss/actor"])
                    entropy.append(loss["entropy"])
                    alpha_loss.append(loss["loss/alpha"])
                    t.set_postfix(**loss)
                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    num_timesteps += 1
                    t.update(1)
            # evaluate current policy
            # if e % 50 == 0:
            #     if self.eval_env.id == 'Abiomed-v0':
            #         eval_info, _ = self.evaluate()
            #         ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            #         ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            #         ep_accuracy_mean, ep_accuracy_std = np.mean(eval_info["eval/episode_accuracy"]), np.std(eval_info["eval/episode_accuracy"])
            #         ep_1_off_accuracy_mean, ep_1_off_accuracy_std = np.mean(eval_info["eval/episode_1_off_accuracy"]), np.std(eval_info["eval/episode_1_off_accuracy"])
            #         ep_mse_mean, ep_mse_std = np.mean(eval_info["eval/mse"]), np.std(eval_info["eval/mse"])
            #         acc_l.append(ep_accuracy_mean)
            #         off_acc.append(ep_1_off_accuracy_mean)
            #         acc_std_l.append(ep_accuracy_std)
            #         off_acc_std.append(ep_1_off_accuracy_std)
            #         self.logger.record("eval/episode_accuracy", ep_accuracy_mean, num_timesteps, printed=False)
            #         self.logger.record("eval/episode_1_off_accuracy", ep_1_off_accuracy_mean, num_timesteps, printed=False)
            #         self.logger.record("eval/mse", ep_mse_mean, num_timesteps, printed=False)
            #         self.logger.print(f"Epoch #{e}: episode_accuracy: {ep_accuracy_mean:.3f} ± {ep_accuracy_std:.3f},\
            #                         episode_1_off_accuracy: {ep_1_off_accuracy_mean:.3f} ± {ep_1_off_accuracy_std:.3f}"
            #                         )
            #     else:
            #         eval_info,_ = self._evaluate()
                    
            #         ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            #         ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                
                    
            #     reward_l.append(ep_reward_mean)
            #     reward_std_l.append(ep_reward_std)
                
            #     self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
            #     self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)

            #     self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
            #                     episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}"
            #                     )
            
            # save policy
            model_save_dir = util.logger_model.log_path
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(self.algo.policy.to('cpu').state_dict(), os.path.join(model_save_dir, f"policy_{self.env_name}_{self.iter}.pth")) #plot q_values for each epoch
            print(util.device)
            self.algo.policy.to(util.device)        
        if self.run_id != 0: 
            plot_q_value(np.array(q1_l).reshape(-1,1), 'Q1')
            plot_q_value(np.array(q2_l).reshape(-1,1), 'Q2')
            plot_q_value(np.array(q_l).reshape(-1,1), 'Q')

            plot_p_loss(np.array(critic_loss1).reshape(-1,1), 'Critic1')
            plot_p_loss(np.array(critic_loss2).reshape(-1,1), 'Critic2')
            plot_p_loss(np.array(actor_loss).reshape(-1,1), 'Actor')
            plot_p_loss(np.array(entropy).reshape(-1,1), 'Entropy')
            plot_p_loss(np.array(alpha_loss).reshape(-1,1), 'Alpha')

            plot_accuracy(np.array(reward_l), np.array(reward_std_l)/self._eval_episodes, 'Average Return')
            plot_accuracy(np.array(acc_l), np.array(acc_std_l)/self._eval_episodes, 'Accuracy')
            plot_accuracy(np.array(off_acc), np.array(off_acc_std)/self._eval_episodes, '1-off Accuracy')

        # self.algo.policy.to(util.device)
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))
        


    def _evaluate(self, world_model, crps_scale = 0):

        #TODO: in progress
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        obs_ = []
        next_obs_ = []
        action_ = []
        full_action_ = []
        reward_ = []
        terminal_ = []
        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action) #d4rl env
            #predict next_obs
            pred_input = torch.FloatTensor(np.concatenate([obs, action])).to(world_model.device)

            next_obs_mult = world_model.predict_multiple(pred_input, num_samples=50).cpu().numpy()
            #calculate std over next_obs_mult
            std = (next_obs_mult.std(axis=0)).mean().item()
            penalized_reward = reward - crps_scale * std

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

            obs_.append(list(obs[0]))
            next_obs_.append(list(next_obs_mult.mean(axis=0).reshape(obs.shape)[0]))
            action_.append(action)
            reward_.append(penalized_reward)
            terminal_.append(terminal)
        dataset = {
                'observations': np.array(obs_),
                'actions': np.array(action_),  # Reshape to ensure it's 2D
                'rewards': np.array(reward_),
                'terminals': np.array(terminal_),
                'next_observations': np.array(next_obs_),
            }
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        },dataset


    def evaluate(self):
        self.algo.policy.eval()
        obs = self.eval_env.reset().reshape(1,-1)
        
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
        raw_reward_ = []
        terminal_counter = 0
        N = self.eval_env.data['observations'].shape[0]

        indx = np.random.choice(N,
                                size=self._eval_episodes,
                                replace=False)
        
        for i in tqdm(range(indx.shape[0])):
            start_time = time.time()
            act = self.eval_env.get_pl()
           
            next_state_gt = self.eval_env.get_next_obs()
            action = self.algo.policy.sample_action(obs, deterministic=True)
            action = action.repeat(90) #repeat the action for 90 steps

            full_pl = self.eval_env.get_full_pl()
            next_obs, reward, terminal, info = self.eval_env.step_std(action) #next state predictions
            rwd_rewards = self.eval_env.compute_reward(next_obs) #dropout not enabled prediction
            next_obs = self.eval_env.normalize(next_obs, idx=np.arange(0,12))
            #MSE of next_obs and next_state_gt
            mse = np.mean((next_obs.reshape(-1,1) - next_state_gt)**2)*self.eval_env.rwd_stds[12]

            #obs: (0,90) next_state_gt:(90,180) next_obs: (90,180), action: (90,180) act: (90,180)
            if info != {}:
                crps_list.extend([info['std']])
            episode_reward += reward
            episode_length += 1

            terminal_counter += 1
            acc, acc_1_off = self.eval_bcq(action, full_pl)
            # acc_total += acc
            # acc_1_off_total += acc_1_off

            if i == indx.shape[0]-1:
                self.plot_predictions_rl(obs.reshape(1,90,12), next_state_gt.reshape(1,90,12), next_obs.reshape(1,90,12), action.reshape(1,90), full_pl.reshape(1,90), num_episodes)
            
            
            if terminal_counter == self.terminal_counter:

                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward,
                      "episode_length": episode_length,
                        "episode_accuracy": acc, 
                        "episode_1_off_accuracy": acc_1_off,
                        'mse': mse,
                        }
                )
                num_episodes +=1
                terminal_counter = 0
                episode_reward, episode_length = 0, 0
                
                # self.logger.print("EVAL TIME: {:.3f}s".format(time.time() - start_time))
            #obs, next_obs, reward, done

            obs_.append(list(obs[0]))
            next_obs_.append(list(next_obs.reshape(obs.shape)[0]))
            action_.append(action[0])
            full_action_.append(full_pl)
            reward_.append(reward)
            terminal_.append(terminal)

            raw_reward_.append(rwd_rewards)
            if num_episodes != self.eval_env.data['observations'].shape[0]:
                obs = self.eval_env.get_obs().reshape(1,-1)
            else:
                break
        
        #rewards withput std penalty using D_1
        with open(f'/data/abiomed_tmp/intermediate_data_uambpo/raw_rewards_{self.eval_env.crps_scale}_{self.iter}.pkl', 'wb') as f:
            np.save(f, np.array(raw_reward_))
        #rewards with penalty using D_1
        with open(f'/data/abiomed_tmp/intermediate_data_uambpo/rewards_{self.eval_env.crps_scale}_{self.iter}.pkl', 'wb') as f:
            np.save(f, np.array(reward_))
        #save crps list
        with open(f'/data/abiomed_tmp/intermediate_data_uambpo/crps_list_{self.eval_env.crps_scale}_{self.iter}.pkl', 'wb') as f:
            np.save(f, np.array(crps_list))
        


        

        #need actions to be unnormalized for plotting
        action_ = self.eval_env.unnormalize(np.array(action_), idx=12)
        full_action_ = self.eval_env.unnormalize(np.array(full_action_), idx=12).reshape(-1,90)
        dataset = {
                'observations': np.array(obs_),
                'actions': np.array(action_).reshape(-1, 1),  # Reshape to ensure it's 2D
                'rewards': np.array(reward_),
                'terminals': np.array(terminal_),
                'next_observations': np.array(next_obs_),
                'full_actions': np.array(full_action_),  # Reshape to ensure it's 2D
            }
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_accuracy": [ep_info["episode_accuracy"] for ep_info in eval_ep_info_buffer],
            "eval/episode_1_off_accuracy": [ep_info["episode_1_off_accuracy"] for ep_info in eval_ep_info_buffer],
            "eval/mse": [ep_info["mse"] for ep_info in eval_ep_info_buffer],
        }, dataset


    def plot_predictions_rl(self, src, tgt_full, pred, pl, pred_pl,iter):

    
        input_color = 'tab:blue'
        pred_color = 'tab:orange' #label="input",
        gt_color = 'tab:green'
        rl_color = 'tab:red'

        fig, ax1 = plt.subplots(figsize = (8,5.8), dpi=300)
                                        
        default_x_ticks = range(0, 181, 18)
        x_ticks = np.array(list(range(0, 31, 3)))
        plt.xticks(default_x_ticks, x_ticks)

        ax1.axvline(x=90, linestyle='--', c='black', alpha =0.7)

        plt.plot(range(90), self.eval_env.unnormalize(src.reshape(90,12)[:,0], idx = 0), color=input_color)
        plt.plot(range(90,180), self.eval_env.unnormalize(tgt_full.reshape(90,12)[:,0], idx = 0), label ="ground truth MAP", color=input_color)
        plt.plot(range(90,180), self.eval_env.unnormalize(pred.reshape(90,12)[:,0], idx = 0),  label ='prediction MAP', color=pred_color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(range(90,180), self.eval_env.unnormalize(pl.reshape(-1,1), idx = 12)*1000,'--',label ='ground truth PL', color=gt_color)
        ax2.plot(range(90,180), self.eval_env.unnormalize(pred_pl.reshape(-1,1), idx = 12)*1000,'--',label ='BCQ PL', color=rl_color)
        ax2.set_ylim((500,10000))
        ax1.legend(loc=3)
        ax2.legend(loc=1)

        ax1.set_ylabel('MAP (mmHg)',  fontsize=20)
        ax2.set_ylabel('Pump Speed',  fontsize=20)
        ax1.set_xlabel('Time (min)', fontsize=20)
        ax1.set_title(f"MAP Prediction and P-level")
        wandb.log({f"plot_batch_{iter}": wandb.Image(fig)})

        plt.show()

        
    def eval_bcq(self, y_pred_test, y_test):


        pred_unreg =  self.eval_env.unnormalize(np.array(y_pred_test).reshape(-1,1), idx=12)
        real_unreg = self.eval_env.unnormalize(np.array(y_test).reshape(-1,1), idx=12) 


        pl_pred_fl = np.round(pred_unreg.flatten())
        pl_true_fl = np.round(real_unreg.flatten())
        n = len(pl_pred_fl)


        accuracy = sum(pl_pred_fl == pl_true_fl)/n
        accuracy_1_off = (sum(pl_pred_fl == pl_true_fl) + sum(pl_pred_fl+1 == pl_true_fl)+sum(pl_pred_fl-1 == pl_true_fl))/n

        return accuracy, accuracy_1_off
