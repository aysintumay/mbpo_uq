import gym
from gym import spaces
import numpy as np
import torch
from models.world_transformer import WorldTransformer
from tqdm import tqdm
import scoring

class AbiomedEnv(gym.Env):
    def __init__(self, args, logger,data_name, replay = False,  offline_buffer = None, pretrained = False):
        super(AbiomedEnv, self).__init__()
        # Replace obs_dim and action_dim with actual dimensions
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12*90,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.id = 'Abiomed-v0'
        self.replay = replay
        self.pretrained = args.pretrained
        self.offline_buffer = offline_buffer
        self.logger = logger
        self.args = args
        self.data_name = data_name
        self.world_model = WorldTransformer(args = self.args, logger = self.logger, pretrained = self.pretrained)
        # self.trained_world_model = self.world_model.load_model()
        self.data = self.load_data()
        self.current_index = 0
        self.rwd_means = []
        self.rwd_stds = []

    def load_data(self, offline_buffer=None):


        def generate_buffer(data, length=90):

            #dont take p-level in the state
            state_dim = length * (data.shape[-1]-1)
            #buffer = StandardBuffer(state_dim, 12, 1e6, 'cpu',action_size=length)
        
            obs = []
            next_obs = []
            action_l = []
            reward_l = []
            done_l = []
            full_action_l = []

            norm_data = (data - self.rwd_means) / self.rwd_stds
            
            for i in tqdm(range(norm_data.shape[0])):
                #try adding pump speed to the state
                row = norm_data[i]
                unnorm_row = data[i]
                reward = self.compute_reward(unnorm_row[90:, :12])
                reward = reward/31
                state = row[:90, :12].flatten()
                next_state = row[90:, :12].flatten()

                #take p-level as action
                action = row[90:, 12].mean()
                all_action =  row[90:, 12]
                done = 0

                if np.isnan(state).any() or np.isnan(next_state).any():
                    continue  
                # append to each arry
                obs.append(state)
                next_obs.append(next_state)
                action_l.append(action)
                reward_l.append(reward)
                done_l.append(done)
                full_action_l.append(all_action)  # Store the full action for analysis

                    
            return {
                    'observations': np.array(obs),
                    'actions': np.array(action_l).reshape(-1, 1),  # Reshape to ensure it's 2D
                    'rewards': np.array(reward_l),
                    'terminals': np.array(done_l),
                    'next_observations': np.array(next_obs),
                    'full_actions': np.array(full_action_l)  # Store the full action for analysis
                    }
    
        if offline_buffer is None:
            #change to the formal path later
            train = torch.load(f"/data/abiomed_tmp/processed/pp_{self.data_name}_amicgs.pt").numpy()
            # test = torch.load(f"/data/abiomed_tmp/processed/pp_test_amicgs.pt").numpy()

            #dont take ID column
            train = train[: ,:, :-1]
            self.rwd_means = train.mean(axis=(0, 1))
            self.rwd_stds = train.std(axis=(0, 1))

            train_dict = generate_buffer(train)
            return train_dict
        else:
            return offline_buffer
    
    def get_dataset(self):

        return self.load_data()
     

    def reset(self):
        self.current_index = 0
        return self.data['observations'][self.current_index]
    
    def get_pl(self):
        return self.data['actions'][self.current_index]
    
    def get_full_pl(self):
        return self.data['full_actions'][self.current_index]
    
    def get_next_obs(self):
        return self.data['next_observations'][self.current_index]
    
    def get_obs(self):
        return self.data['observations'][self.current_index]
    
    def unnormalize(self, data, idx):
         return data  * self.rwd_stds[idx] +  self.rwd_means[idx]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, logging, and sometimes learning)
        """


        if self.current_index%2 == 0:
            obs = self.data['observations'][self.current_index]
            next_state = self.data['next_observations'][self.current_index]
        else:
            obs = self.data['next_observations'][self.current_index-1]
            next_state = self.data['observations'][self.current_index]
            
        dataloder = self.world_model.resize(obs, action, next_state)
        next_obs = self.world_model.predict(dataloder)
        num_samples = 50
        output_mult = self.world_model.model.sample_autoregressive_multiple(torch.Tensor(obs.reshape(-1, 90, self.args.seq_dim)).to(self.args.device), torch.Tensor(action.reshape(1,-1)).to(self.args.device), num_samples=num_samples,)
        # output_mult_all.append(output_mult)
        next_obs_unnorm = self.unnormalize(next_obs, np.arange(0,12))
        output_mult = output_mult.detach().cpu().numpy()
        for i in range(num_samples):
            output_mult[i] = self.unnormalize(output_mult[i], np.arange(0,12))
        #get rewards
        reward, crps = self.get_reward_crps(next_obs_unnorm, output_mult, next_state.reshape(1, 90, self.args.seq_dim))

        # reward = self.compute_reward(next_obs_unnorm)
        #unnormalize
        done = self.check_terminal_condition()
        info = {}
        self.current_index += 1
        return next_obs, reward, done, crps

    def compute_reward(self, data, crps=None, map_dim = 0, pulsat_dim = 7, hr_dim=9, lvedp_dim=4):
        ## GETS rid of min map- redundant
        score = 0
        ### A HIGH SCORE IS BAD
        ### A LOW SCORE IS GOOD
        
        ## MAP ## 
        map_val = data[..., map_dim]
        if map_val.any() >=60.:
            score+=0
        elif (map_val.any() >=50.) & (map_val.any()<=59.):
            score+=1
        elif (map_val.any() >=40.) & (map_val.any()<=49.):
            score+=3
        else: score+=7 # <40 mmHg

        ## TIME IN HYPOTENSION ##
        ts_hypo = np.mean(map_val<60) * 100
        if ts_hypo<0.1:
            score+=0
        elif (ts_hypo >=0.1) & (ts_hypo <=0.2):
            score+=1
        elif (ts_hypo >0.2) & (ts_hypo <=0.5):
            score+=3
        else: score+=7 # greater than 50%
        
        ## Pulsatility ##
        puls = data[..., pulsat_dim]
        if puls.any() > 20.:
            score+=0
        elif (puls.any() <=20.) & (puls.any() >10.):
            score+=5
        else: score+=7 # puls <= 10

        ## Heart Rate ## 
        hr = data[..., hr_dim]
        if hr.any() >= 100.: #tachycardic
            score+=3
        if hr.any() <=50.: # bradycardic
            score+=3

        ## LVEDP ##
        lvedp = data[..., lvedp_dim]
        if lvedp.any() > 20.:
            score+=7
        if (lvedp.any() >=15.) & (hr.any() <=20.):
            score += 4
        
        """
        ## CPO ##
        if cpo > 1.: 
            score+=0
        elif (cpo > 0.6) & (cpo <=1.):
            score+=1
        elif (cpo > 0.5) & (cpo <=0.6):
            score+=3
        else: score+=5 # cpo <=0.5
        """
        if crps:
            score = score - 0.01*crps

        
        return -score


    def compute_reward_smooth(data, map_dim=0, pulsat_dim=6, hr_dim=7, lvedp_dim=3, crps = None):
        '''
        Differentiable version of the reward function using PyTorch
        '''
        score = torch.tensor(0.0, device=data.device)
        relu = torch.nn.ReLU()
        # MAP component
        map_data = data[..., map_dim]

        # MinMAP range component
        minMAP = torch.min(map_data)
        score += relu(7 * (60 - minMAP) / 20) #relu(7 * (60 - minMAP) / 20)  # Linear score from 0 at MAP=60 to 7 at MAP=40 and below
        
        # # Time MAP < 60 component
        # time_below_60 = torch.mean(smooth_threshold(-map_data, -60)) * 100
        # score += relu(7/5 * time_below_60)

        # Heart Rate component
        hr = torch.min(data[..., hr_dim])
        # Polynomial penalty for heart rate outside 50-100 range
        hr_penalty = 3 * (hr - 75)**2 / 625 # Quadratic penalty centered at hr=75, max penalty at hr=50 or 100
        score += hr_penalty
        
        # Pulsatility component
        pulsat = torch.min(data[..., pulsat_dim])
        pulsat_penalty = 7 * (20 - pulsat) / 20
        score += pulsat_penalty
        
        if crps:
            score = score - crps
        return -score


    def get_reward_crps(self, data, mult_pred, target):
        
        '''
        data: next state prediction ground truth (1,90,12)
        mult_pred: (50,1,90,12)
        target: next state (1,90,12)

        return: 1 crps and 1 reward value for each (1,90,12) sample
        '''
        
        
        # rewards = []
        # map_only = []
        # crps_ts = []

        for i in range(data.shape[0]):
            crps = scoring.crps_evaluation(np.array(mult_pred[i,:,:,0]), np.array(target[i,:,0]))
            # crps_ts.extend(crps)
            # un = scoring.unnorm_all(data[i], self.rwd_stds, self.rwd_means)
            r1 = self.compute_reward(np.array(data[i]), crps.mean())
            # rewards.append(r1)
            # cum_rewards = np.mean(rewards)
        return r1, crps 



    def check_terminal_condition(self):
        # Define when an episode should end
        return self.current_index >= len(self.data['observations']) - 1
