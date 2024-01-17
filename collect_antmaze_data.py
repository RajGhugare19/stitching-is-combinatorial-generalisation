import os
import math
import torch
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from utils import get_maze_map, AntmazeWrapper

'''
Code taken from https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation/blob/main/scripts/antmaze/create_antmaze_dataset.py
'''

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

EXPLORATION_ACTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

class QIteration:
    """Solves for optimal policy with Q-Value Iteration.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/q_iteration.py
    """
    def __init__(self, maze):
        self.maze = maze
        self.num_states = maze.map_length * maze.map_width
        self.num_actions = len(EXPLORATION_ACTIONS.keys())
        self.rew_matrix = np.zeros((self.num_states, self.num_actions))
        self.compute_transition_matrix()

    def generate_path(self, current_cell, goal_cell):
        self.compute_reward_matrix(goal_cell)
        q_values = self.get_q_values()
        current_state = self.cell_to_state(current_cell)
        waypoints = {}
        while True:
            action_id = np.argmax(q_values[current_state])
            next_state, _ = self.get_next_state(current_state, EXPLORATION_ACTIONS[action_id])
            current_cell = self.state_to_cell(current_state)
            waypoints[current_cell] = self.state_to_cell(next_state)
            if waypoints[current_cell] == goal_cell:
                break

            current_state = next_state

        return waypoints

    def reward_function(self, desired_cell, current_cell):
        if desired_cell == current_cell:
            return 1.0
        else:
            return 0.0

    def state_to_cell(self, state):
        i = int(state/self.maze.map_width)
        j = state % self.maze.map_width
        return (i, j)

    def cell_to_state(self, cell):
        return cell[0] * self.maze.map_width + cell[1]

    def get_q_values(self, num_itrs=50, discount=0.99):
        q_fn = np.zeros((self.num_states, self.num_actions))
        for _ in range(num_itrs):
            v_fn = np.max(q_fn, axis=1)
            q_fn = self.rew_matrix + discount*self.transition_matrix.dot(v_fn)
        return q_fn

    def compute_reward_matrix(self, goal_cell):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                next_state, _ = self.get_next_state(state, EXPLORATION_ACTIONS[action])
                next_cell = self.state_to_cell(next_state)
                self.rew_matrix[state, action] = self.reward_function(goal_cell, next_cell)

    def compute_transition_matrix(self):
        """Constructs this environment's transition matrix.
        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        self.transition_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))
        for state in range(self.num_states):
            for action_idx, action in EXPLORATION_ACTIONS.items():
                next_state, valid = self.get_next_state(state, action)
                if valid:
                    self.transition_matrix[state, action_idx, next_state] = 1

    def get_next_state(self, state, action):
        cell = self.state_to_cell(state)

        next_cell = tuple(map(lambda i, j: int(i + j), cell, action))
        next_state = self.cell_to_state(next_cell)

        return next_state, self._check_valid_cell(next_cell)

    def _check_valid_cell(self, cell):
        # Out of map bounds
        if cell[0] >= self.maze.map_length:
            return False
        elif cell[1] >= self.maze.map_width:
            return False
        # Wall collision
        elif self.maze.maze_map[cell[0]][cell[1]] == 1:
            return False
        else:
            return True
        
class WaypointController:
    """Generic agent controller to follow waypoints in the maze.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/waypoint_controller.py
    """

    def __init__(
        self, maze, model_callback, waypoint_threshold=0.45
    ):
        self.global_target_xy = np.empty(2)
        self.maze = maze
        self.maze_solver = QIteration(maze=self.maze)

        self.model_callback = model_callback
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = None

    def compute_action(self, obs):
        # Check if we need to generate new waypoint path due to change in global target
        if (
            np.linalg.norm(self.global_target_xy - obs["desired_goal"]) > 1e-3
            or self.waypoint_targets is None
        ):
            # Convert xy to cell id
            achived_goal_cell = tuple(self.maze.cell_xy_to_rowcol(obs["achieved_goal"]))
            self.global_target_id = tuple(
                self.maze.cell_xy_to_rowcol(obs["desired_goal"])
            )
            
            self.global_target_xy = obs["desired_goal"]

            self.waypoint_targets = self.maze_solver.generate_path(
                achived_goal_cell, self.global_target_id
            )
            # Check if the waypoint dictionary is empty
            # If empty then the ball is already in the target cell location
            if self.waypoint_targets:
                self.current_control_target_id = self.waypoint_targets[
                    achived_goal_cell
                ]
                # If target is global goal go directly to goal position
                if self.current_control_target_id == self.global_target_id:
                    self.current_control_target_xy = obs['desired_goal']
                else:
                    self.current_control_target_xy = self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    ) - np.random.uniform(size=(2,)) * 0.1
            else:
                self.waypoint_targets[
                    self.current_control_target_id
                ] = self.current_control_target_id
                self.current_control_target_id = self.global_target_id
                self.current_control_target_xy = self.global_target_xy

        # Check if we need to go to the next waypoint
        dist = np.linalg.norm(self.current_control_target_xy - obs["achieved_goal"])
        if (
            dist <= self.waypoint_threshold
            and self.current_control_target_id != self.global_target_id
        ):
            self.current_control_target_id = self.waypoint_targets[
                self.current_control_target_id
            ]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:

                self.current_control_target_xy = obs['desired_goal']
            else:
                self.current_control_target_xy = (
                    self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    )
                    - np.random.uniform(size=(2,)) * 0.1
                )  
        action = self.model_callback(obs, self.current_control_target_xy)
        return action

def wrap_maze_obs(obs, waypoint_xy):
    """Wrap the maze obs into one suitable for GoalReachAnt."""
    goal_direction = waypoint_xy - obs["achieved_goal"]
    observation = np.concatenate([obs["observation"][2:], goal_direction])
    return observation

def get_start_state_goal_pairs(dataset_name):

    if dataset_name == "antmaze-large-sl":
        train_start_state_goal = [
            {"goal_cells": np.array([ [5,4], [1,10], [7,10] ], dtype=np.int32),
            "reset_cells": np.array([ [7,1] ], dtype=np.int32),
            },
        ]
        
    elif dataset_name == "antmaze-large-v0":
        train_start_state_goal = [
            {"goal_cells": np.array([ [1,1], [1,2], [1,3], [1,4], [2,1], [2,4], [3,1], [3,2], [3,3], [3,4], [4,1], [5,1], [5,2], [6,2], [7,1], [7,2] ], dtype=np.int32),
            "reset_cells": np.array([ [1,1], [1,2], [1,3], [1,4], [2,1], [2,4], [3,1], [3,2], [3,3], [3,4], [4,1], [5,1], [5,2], [6,2], [7,1], [7,2] ], dtype=np.int32),
            },

            {"goal_cells": np.array([ [3,2], [3,3], [3,4], [3,5], [1,6], [2,6], [3,6], [4,6], [5,6], [6,6], [7,6], [7,5], [7,4], [6,4], [5,4], [1,6], [2,6], [1,7], [1,8], [1,9], [1,10], [2,10], [3,10], [3,9], [3,8], [2,8], [4,10], [5,10], [5,9], [5,8], [5,7], [4,6], [5,6], [6,8], [7,8], [7,9], [7,10] ], dtype=np.int32),
            "reset_cells": np.array([ [3,2], [3,3], [3,4], [3,5], [1,6], [2,6], [3,6], [4,6], [5,6], [6,6], [7,6], [7,5], [7,4], [6,4], [5,4], [1,7], [1,8], [1,9], [1,10], [2,10], [3,10], [3,9], [3,8], [2,8], [4,10], [5,10], [5,9], [5,8], [5,7], [6,8], [7,8], [7,9], [7,10] ], dtype=np.int32),
            },
        ]

    elif dataset_name == 'antmaze-medium-v0':
        train_start_state_goal = [
            {'goal_cells': np.array([4,6], dtype=np.int32),
            'reset_cells': np.array([6,5], dtype=np.int32),
            'name' : 'bottom_rightish_to_center',
            },

            {'goal_cells': np.array([6,5], dtype=np.int32),
            'reset_cells': np.array([4,6], dtype=np.int32),
            'name' : 'center_to_bottom_rightish',
            },

            {'goal_cells': np.array([4,4], dtype=np.int32),
            'reset_cells': np.array([6,1], dtype=np.int32),
            'name' : 'bottom_left_to_center',
            },

            {'goal_cells': np.array([6,1], dtype=np.int32),
            'reset_cells': np.array([4,4], dtype=np.int32),
            'name' : 'center_to_bottom_left',
            },

            {'goal_cells': np.array([4,2], dtype=np.int32),
            'reset_cells': np.array([6,1], dtype=np.int32),
            'name' : 'bottom_left_to_center_leftish',
            },

            {'goal_cells': np.array([6,1], dtype=np.int32),
            'reset_cells': np.array([4,2], dtype=np.int32),
            'name' : 'center_leftish_to_bottom_left',
            },

            {'goal_cells': np.array([4,6], dtype=np.int32),
            'reset_cells': np.array([1,6], dtype=np.int32),
            'name' : 'top_right_to_center',
            },

            {'goal_cells': np.array([1,6], dtype=np.int32),
            'reset_cells': np.array([4,6], dtype=np.int32),
            'name' : 'center_to_top_right',
            },
        ]

    elif dataset_name == "antmaze-umaze-v0":
        train_start_state_goal = [    
            {"goal_cells": np.array([ [1,1], [1,2], [1,3] ], dtype=np.int32),
            "reset_cells": np.array([ [1,1], [1,2], [1,3] ], dtype=np.int32),
            },

            {"goal_cells": np.array([ [3,1], [3,2], [3,3] ], dtype=np.int32),
            "reset_cells": np.array([ [3,1], [3,2], [3,3] ], dtype=np.int32),
            },

            {"goal_cells": np.array([ [1,3], [2,3] ], dtype=np.int32),
            "reset_cells": np.array([ [1,3], [2,3] ], dtype=np.int32),
            },

            {"goal_cells": np.array([ [3,3], [2,3] ], dtype=np.int32),
            "reset_cells": np.array([ [3,3], [2,3] ], dtype=np.int32),
            },
        ]
    
    else:
        raise NotImplementedError
    
    return train_start_state_goal

def load_policy(policy_file):
    data = torch.load(policy_file)
    policy = data['exploration/policy'].to('cpu')
    env = data['evaluation/env']
    print("Policy loaded")
    return policy, env

def collect_dataset(dataset_name, num_data):
    if os.path.isfile("data/"+dataset_name+".pkl"):
        print("A dataset with the same name already exists. Doing nothing. Delete the file if you want any changes.")
        exit()

    if "antmaze-umaze" in dataset_name:
        env_name = "AntMaze_UMaze-v4"
    elif "antmaze-medium" in dataset_name:
        env_name = "AntMaze_Medium-v4"
    elif "antmaze-large" in dataset_name:
        env_name = "AntMaze_Large-v4"
    else:
        raise NotImplementedError
    
    # environment initialisation
    env = AntmazeWrapper(gym.make(env_name, continuing_task=False))#, render_mode="human"))

    # data placeholders
    observation_data = {
        "episode_id": np.zeros(shape=(int(num_data),), dtype=np.int32),
        "observation" : np.zeros(shape=(int(num_data), *env.observation_space["observation"].shape), dtype=np.float32),
        "achieved_goal" : np.zeros(shape=(int(num_data), *env.observation_space["achieved_goal"].shape), dtype=np.float32),
    }
    action_data = np.zeros(shape=(int(num_data), *env.action_space.shape), dtype=np.float32)
    termination_data = np.zeros(shape = (int(num_data),), dtype=bool)

    # get data collecting start state and goal pairs
    train_start_state_goal = get_start_state_goal_pairs(dataset_name)

    # controller
    model = SAC.load('GoalReachAnt_model.zip')    
    def action_callback(obs, waypoint_xy):
        return model.predict(wrap_maze_obs(obs, waypoint_xy))[0]
    waypoint_controller = WaypointController(env.maze, action_callback)

    terminated = False
    truncated = False
    data_idx = 0
    episode_idx = 0
    success_count = 0

    if train_start_state_goal is not None:
        num_dc = len(train_start_state_goal)
        sg_dict = train_start_state_goal[ episode_idx % num_dc ]
        if len(sg_dict["goal_cells"].shape) == 2:
            x,_ = sg_dict["goal_cells"].shape
            id = np.random.choice(x, size=1)[0]
            sg_dict["goal_cell"] = sg_dict["goal_cells"][id]
        else:
            sg_dict["goal_cell"] = sg_dict["goal_cells"]
        
        if len(sg_dict["reset_cells"].shape) == 2:
            x,_ = sg_dict["reset_cells"].shape
            id = np.random.choice(x, size=1)[0]
            sg_dict["reset_cell"] = sg_dict["reset_cells"][id]
        else:
            sg_dict["reset_cell"] = sg_dict["reset_cells"]
        obs, _ = env.reset(options=sg_dict)
    else:
        obs, _ = env.reset()
    
    episode_start_idx = 0
    while data_idx < int(num_data)-1:
  
        action = waypoint_controller.compute_action(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high, dtype=np.float32)

        observation_data["episode_id"][data_idx] = episode_idx
        observation_data["observation"][data_idx] = obs["observation"]
        observation_data["achieved_goal"][data_idx] = obs["achieved_goal"]

        action_data[data_idx] = action
        termination_data[data_idx] = terminated or truncated
        data_idx += 1

        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if info['success']:
                success_count += 1

                observation_data["episode_id"][data_idx] = episode_idx
                observation_data["observation"][data_idx] = obs["observation"]
                observation_data["achieved_goal"][data_idx] = obs["achieved_goal"]
                termination_data[data_idx] = terminated or truncated

                data_idx += 1
                episode_idx += 1
                episode_start_idx = data_idx
        
            else:
                data_idx = episode_start_idx

            if train_start_state_goal is not None:
                sg_dict = train_start_state_goal[ episode_idx % num_dc ]
                if len(sg_dict["goal_cells"].shape) == 2:
                    x,_ = sg_dict["goal_cells"].shape
                    id = np.random.choice(x, size=1)[0]
                    sg_dict["goal_cell"] = sg_dict["goal_cells"][id]
                else:
                    sg_dict["goal_cell"] = sg_dict["goal_cells"]
                
                if len(sg_dict["reset_cells"].shape) == 2:
                    x,_ = sg_dict["reset_cells"].shape
                    id = np.random.choice(x, size=1)[0]
                    sg_dict["reset_cell"] = sg_dict["reset_cells"][id]
                else:
                    sg_dict["reset_cell"] = sg_dict["reset_cells"]
                obs, _ = env.reset(options=sg_dict)
            else:
                obs, _ = env.reset()

            terminated = False
            truncated = False

        if (data_idx + 1) % (num_data // 20) == 0:
            print("STEPS RECORDED:")
            print(data_idx)

            print("EPISODES RECORDED:")
            print(episode_idx)

            print("SUCCESS EPISODES RECORDED:")
            print(success_count)


        dataset = {"observations": observation_data, 
                "actions": action_data, 
                "terminations": termination_data,
                "success_count": success_count,
                "episode_count": episode_idx,
                }
    
    with open("data/"+dataset_name+".pkl", "wb") as fp:
        pickle.dump(dataset, fp)
            
if __name__ == "__main__":
    import sys
  
    dataset_name = sys.argv[1]

    seed = sys.argv[2]
    num_data = int(1e6)

    np.random.seed(int(seed))
    collect_dataset(dataset_name, num_data)