
import torch
import numpy as np
import pickle
from models.d4rl_world_model import D4RLWorldModel, MLPNetwork
from tqdm import tqdm
import argparse
from common.normalizer import StandardNormalizer
from common import util
import scoring


model_paths = {
    'halfcheetah-random-v0': "saved_models/halfcheetah-random-v0/world_model_0.55.pth",
    'halfcheetah-expert-v0': "saved_models/halfcheetah-expert-v0/world_model_0.63.pth",
    'walker2d-random-v0': "saved_models/walker2d-expert-v0/world_model_0.86.pth",
    'walker2d-expert-v0': "saved_models/walker2d-expert-v0/world_model_0.57.pth",
}


def get_crps_list(env_name, discount_factor):
    print(f"Starting CRPS list for {env_name} with discount {discount_factor}")
    model = D4RLWorldModel(env_name)
    model_path = model_paths[env_name]
    model.load_model(model_path)

    dataset = model.dataset

    crps_list = []
    batch_size = 50
    total_batches = len(dataset['observations']) // batch_size
    num_samples = 50

    with torch.no_grad():
        for i in tqdm(range(total_batches)):
            state = dataset['observations'][i*batch_size:(i+1)*batch_size]
            action = dataset['actions'][i*batch_size:(i+1)*batch_size]
            next_state = dataset['next_observations'][i*batch_size:(i+1)*batch_size]

            states = np.repeat(state, num_samples, axis=0)
            actions = np.repeat(action, num_samples, axis=0)

            pred_input = torch.FloatTensor(np.concatenate([states, actions], axis=1)).to(model.device)
            model_pred = model.model(pred_input).cpu().numpy()

            for i in range(batch_size):
                crps = scoring.crps_evaluation(model_pred[i * num_samples:(i+1) * num_samples], next_state[i])
                crps_list.append(crps.mean())
    
    crps_list = np.array(crps_list)
    print(crps_list.shape)

    # verify crps_list is the same length as the dataset
    assert len(crps_list) == len(dataset['observations']), "crps_list is not the same length as the dataset"
    
    np.save(f'/data/abiomed_tmp/intermediate_data_d4rl/{env_name}_crps_list.npy', crps_list)


    discounted_dataset = {
        'observations': dataset['observations'],
        'actions': dataset['actions'],
        'rewards': dataset['rewards'] - discount_factor * crps_list,
        'terminals': dataset['terminals'],
        'next_observations': dataset['next_observations'],
        'actions': dataset['actions'], 
    }

    with open(f"/data/abiomed_tmp/intermediate_data_d4rl/{env_name}_crps_discount_{discount_factor}.pkl", "wb") as f:
        pickle.dump(discounted_dataset, f)
    
    print(f"Saved discounted dataset for {env_name} with discount {discount_factor}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--discount", type=float, default=0.1)
    args = parser.parse_args()
    
    if args.env_name == "":
        envs = ["halfcheetah-random-v0", "halfcheetah-expert-v0", "walker2d-random-v0", "walker2d-expert-v0"]
        for env in envs:
            get_crps_list(env, args.discount)
    else:
        get_crps_list(args.env_name, args.discount)
