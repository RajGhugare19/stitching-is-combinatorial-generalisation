import os
import torch
import pickle
import minari
import numpy as np
import gymnasium as gym
from datetime import datetime
from sklearn.cluster import KMeans

from collections import defaultdict
from torch.utils.data import Dataset

def convert_remote_to_local(dataset_name, env):

    if os.path.isfile('data/'+dataset_name+'-remote.pkl'):
        print("A dataset with the same name already exists. Using that dataset.") 
        return
    
    if "pointmaze" in dataset_name:
        if "pointmaze-umaze" in dataset_name:
            env_name = 'PointMaze_UMaze-v3'
        elif "pointmaze-medium" in dataset_name:
            env_name = 'PointMaze_Medium-v3'
        elif "pointmaze-large" in dataset_name:
            env_name = 'PointMaze_Large-v3'

        env = gym.make(env_name, continuing_task=False)

    else:
        raise NotImplementedError
    
    minari_dataset = minari.load_dataset(dataset_name)

    # data placeholders
    observation_data = {
        'episode_id': np.zeros(shape=(int(1e6),), dtype=np.int32),
        'observation' : np.zeros(shape=(int(1e6), *env.observation_space['observation'].shape), dtype=np.float32),
        'achieved_goal' : np.zeros(shape=(int(1e6), *env.observation_space['achieved_goal'].shape), dtype=np.float32),
    }
    action_data = np.zeros(shape=(int(1e6), *env.action_space.shape), dtype=np.float32)
    termination_data = np.zeros(shape = (int(1e6),), dtype=bool)

    data_idx = 0
    episode_idx = 0
    for episode in minari_dataset:
        if data_idx + episode.total_timesteps + 1 > int(1e6):

            observation_data['episode_id'][data_idx: ] = episode_idx
            observation_data['observation'][data_idx: ] = episode.observations['observation'][:int(1e6)-data_idx]
            observation_data['achieved_goal'][data_idx: ] = episode.observations['achieved_goal'][:int(1e6)-data_idx]
            
            action_data[data_idx: ] = episode.actions[:int(1e6)-data_idx]
            termination_data[data_idx+1: ] = episode.truncations[:int(1e6)-data_idx-1]

            break

        else:
            try:
                observation_data['episode_id'][data_idx: data_idx+episode.total_timesteps+1] = episode_idx
                observation_data['observation'][data_idx: data_idx+episode.total_timesteps+1] = episode.observations['observation']
                observation_data['achieved_goal'][data_idx: data_idx+episode.total_timesteps+1] = episode.observations['achieved_goal']
                
                action_data[data_idx: data_idx+episode.total_timesteps] = episode.actions
                termination_data[data_idx+1: data_idx+episode.total_timesteps+1] = episode.truncations

                data_idx = data_idx + episode.total_timesteps + 1
                episode_idx = episode_idx + 1

            except:
                # some episodes are wierd; timesteps is equal to num obervations
                continue

        if (episode_idx + 1) % 1000 == 0:
            print('EPISODES RECORDED = ', episode_idx)

        if data_idx >= int(1e6):
            break
        
    print('Total transitions recorded = ', data_idx)

    dataset = {'observations':observation_data, 
            'actions':action_data, 
            'terminations':termination_data,
            }

    with open('data/'+dataset_name+'-remote.pkl', 'wb') as fp:
        pickle.dump(dataset, fp)
    
def extract_discrete_id_to_data_id_map(discrete_goals, dones, last_valid_traj):
    discrete_goal_to_data_idx = defaultdict(list)
    gm = 0
    for i, d_g in enumerate(discrete_goals):

        discrete_goal_to_data_idx[d_g].append(i)
        gm += 1
        
        if (i + 1) % 200000 == 0:
            print('Goals mapped:', gm)
        
        if i == last_valid_traj:
            break
    
    for dg, data_idxes in discrete_goal_to_data_idx.items():
        discrete_goal_to_data_idx[dg] = np.array(data_idxes)

    print('Total goals mapped:', gm)
    return discrete_goal_to_data_idx

def extract_done_markers(dones, episode_ids):
    """Given a per-timestep dones vector, return starts, ends, and lengths of trajs."""

    (ends,) = np.where(dones)
    return ends[ episode_ids[ : ends[-1] + 1 ] ], np.where(1 - dones[: ends[-1] + 1])[0]

class MinariEpisodicTrajectoryDataset(Dataset):
    def __init__(self, dataset_name, remote_data, context_len, augment_data, augment_prob, nclusters=40):
        super().__init__()
        if remote_data:
            path = 'data/'+dataset_name+'-remote.pkl'
        else:
            path = 'data/'+dataset_name+'.pkl'
            
        with open(path, 'rb') as fp:
            self.dataset = pickle.load(fp)

        self.observations = self.dataset['observations']['observation']     
        self.achieved_goals = self.dataset['observations']['achieved_goal']
        self.actions = self.dataset['actions']

        (ends,) = np.where(self.dataset['terminations'])
        self.starts = np.concatenate(([0], ends[:-1] + 1))
        self.lengths = ends - self.starts + 1           #length = number of actions taken in an episode + 1
        self.ends = ends[ self.dataset['observations']['episode_id'][ : ends[-1] + 1 ] ]

        good_idxes = self.lengths > context_len
        print('Throwing away ', np.sum(self.lengths[~good_idxes] - 1), 'number of transitions')
        self.starts = self.starts[good_idxes]           #starts will only contain indices of episodes where number of states > context_len
        self.lengths = self.lengths[good_idxes]
        
        self.num_trajectories = len(self.starts)

        if augment_data:    
            start_time = datetime.now().replace(microsecond=0)
            print('starting kmeans ... ')
            kmeans = KMeans(n_clusters=nclusters, n_init="auto").fit(self.achieved_goals)
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
            print('kmeans done! time taken :', time_elapsed)

            self.discrete_goal_to_data_idx = extract_discrete_id_to_data_id_map(kmeans.labels_, self.dataset['terminations'], self.ends[-1])
            self.achieved_discrete_goals = kmeans.labels_
            kmeans = None
        
        self.state_dim = self.observations.shape[-1]
        self.state_dtype = self.observations.dtype
        self.act_dim = self.actions.shape[-1]
        self.act_dtype = self.actions.dtype
        self.goal_dim = self.achieved_goals.shape[-1]
        self.context_len = context_len
        self.augment_data = augment_data
        self.augment_prob = augment_prob
        self.dataset = None

    def __len__(self):
        return self.num_trajectories * 100
    
    def __getitem__(self, idx):
        '''
        Reminder: np.random.randint samples from the set [low, high)
        '''
        idx = idx % self.num_trajectories
        traj_len = self.lengths[idx] - 1                        #traj_len = T, traj_len is the number of actions taken in the trajectory
        traj_start_i = self.starts[idx]
        assert self.ends[traj_start_i] == traj_start_i + traj_len

        
        if self.augment_data and np.random.uniform(0, 1) <= self.augment_prob:
            correct = False
            while not correct:
                si = traj_start_i + np.random.randint(0, traj_len)          #si can be traj_start_i + [0, T - 1]
                gi = np.random.randint(si, traj_start_i + traj_len) + 1     #gi can be traj_start_i + 1 + [si + 1, T]     
                dummy_discrete_goal = self.achieved_discrete_goals[ gi ]
                nearby_goal_idx = np.random.choice(self.discrete_goal_to_data_idx[dummy_discrete_goal])
                nearby_goal_idx_ends = self.ends[nearby_goal_idx]
                if (gi-si) + (nearby_goal_idx_ends - nearby_goal_idx) + 1 > self.context_len:
                    correct = True
                
            if gi - si < self.context_len:
                goal = torch.tensor(self.achieved_goals[ np.random.randint(nearby_goal_idx + self.context_len - (gi - si), nearby_goal_idx_ends + 1) ]).view(1, -1)
                state = torch.tensor( np.concatenate( [ self.observations[si: gi], self.observations[nearby_goal_idx: nearby_goal_idx + self.context_len - (gi - si)] ] ) )
                action = torch.tensor( np.concatenate( [ self.actions[si: gi], self.actions[nearby_goal_idx: nearby_goal_idx + self.context_len - (gi - si)] ] ) )
            else:
                goal = torch.tensor(self.achieved_goals[ np.random.randint(nearby_goal_idx, nearby_goal_idx_ends + 1) ]).view(1, -1)
                state = torch.tensor(self.observations[si: si + self.context_len])
                action = torch.tensor(self.actions[si: si + self.context_len])

        else:
            si = traj_start_i + np.random.randint(0, traj_len - self.context_len + 1)       #si can be traj_start_i + [0, T-C]  
            gi = np.random.randint(si + self.context_len, traj_start_i + traj_len + 1)      #gi can be [si+C, traj_start_i+T]

            goal = torch.tensor(self.achieved_goals[ gi ]).view(1, -1)
            state = torch.tensor(self.observations[si: si + self.context_len])
            action = torch.tensor(self.actions[si: si + self.context_len])
        
        return state, goal, action

class MinariEpisodicDataset(Dataset):
    def __init__(self, dataset_name, remote_data, augment_data, augment_prob, nclusters=40):
        super().__init__()
        if remote_data:
            path = 'data/'+dataset_name+'-remote.pkl'
        else:
            path = 'data/'+dataset_name+'.pkl'
        
        with open(path, 'rb') as fp:
            self.dataset = pickle.load(fp)

        self.episode_ids = self.dataset['observations']['episode_id']
        self.observations = self.dataset['observations']['observation']
        self.achieved_goals = self.dataset['observations']['achieved_goal']
        self.actions = self.dataset['actions']

        (ends,) = np.where(self.dataset['terminations'])
        self.starts = np.concatenate(([0], ends[:-1] + 1))
        self.lengths = ends - self.starts + 1
        self.num_trajectories = len(self.starts)

        self.ends = ends[ self.dataset['observations']['episode_id'][ : ends[-1] + 1 ] ]
        
        if augment_data:    
            start_time = datetime.now().replace(microsecond=0)
            print('starting kmeans ... ')
            kmeans = KMeans(n_clusters=nclusters, n_init="auto").fit(self.observations)
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
            print('kmeans done! time taken :', time_elapsed)

            self.discrete_goal_to_data_idx = extract_discrete_id_to_data_id_map(kmeans.labels_, self.dataset['terminations'], self.ends[-1])
            self.achieved_discrete_goals = kmeans.labels_
            kmeans = None

        self.goal_dim = self.achieved_goals.shape[-1]
        self.augment_data = augment_data
        self.augment_prob = augment_prob
        self.dataset = None

    def __len__(self):
        return self.num_trajectories * 100
    
    def __getitem__(self, idx):
        '''
        Reminder: np.random.randint samples from the set [low, high)
        '''
        idx = idx % self.num_trajectories
        traj_len = self.lengths[idx] - 1                        #traj_len = T, traj_len is the number of actions taken in the trajectory
        traj_start_i = self.starts[idx]
        assert self.ends[traj_start_i] == traj_start_i + traj_len

        si = np.random.randint(0, traj_len)                     #si can be [0, T-1]  

        state = torch.tensor(self.observations[traj_start_i + si])
        action = torch.tensor(self.actions[traj_start_i + si])
        
        if self.augment_data and np.random.uniform(0, 1) <= self.augment_prob:
            dummy_discrete_goal = self.achieved_discrete_goals[ traj_start_i + np.random.randint(si, traj_len) + 1 ]
            nearby_goal_idx = np.random.choice(self.discrete_goal_to_data_idx[dummy_discrete_goal])            
            goal = torch.tensor(self.achieved_goals[ np.random.randint(nearby_goal_idx, self.ends[nearby_goal_idx] + 1) ])
        else:
            goal = torch.tensor(self.achieved_goals[ traj_start_i + np.random.randint(si, traj_len) + 1 ])

        return state, goal, action