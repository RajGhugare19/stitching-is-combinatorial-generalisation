import numpy as np
import torch.nn as nn
import gymnasium as gym
from typing import Iterable, Any
from gymnasium.spaces import Box
from gymnasium.error import DependencyNotInstalled

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class PreprocessObservationWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    
    def __init__(self, env, shape, grayscale=False) -> None:
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gym.ObservationWrapper.__init__(self, env)

        if isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2 and all(
            x > 0 for x in shape
        ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"
        
        self.grayscale = grayscale
        if grayscale:
             obs_shape = env.observation_space['pixels'].shape
             env.observation_space['pixels'] = Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)

        self.shape = tuple(shape)
        obs_shape = self.shape + env.observation_space['pixels'].shape[2:]
        self.observation_space['pixels'] = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e

        if self.grayscale:
            observation['pixels'] = cv2.cvtColor(observation['pixels'], cv2.COLOR_RGB2GRAY)
        observation['pixels'] = cv2.resize(
            observation['pixels'], self.shape[::-1], interpolation=cv2.INTER_AREA
        ).reshape(self.observation_space['pixels'].shape)
        
        return observation

class AntmazeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """Constructor for the observation wrapper."""
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space['observation'] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_space['observation'].shape[0] + 2,), dtype=np.float64
        )

    def reset(
        self, *, seed=None, options=None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action
    ):
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation):
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        observation['observation'] = np.concatenate((observation['achieved_goal'],  observation['observation']), axis=0)
        return observation

def get_test_start_state_goals(cfg):
    if cfg.dataset_name in ["pointmaze-umaze-v0", "antmaze-umaze-v0"]:
        test_start_state_goal = [
            {'goal_cell': np.array([1,1], dtype=np.int32),
            'reset_cell': np.array([3,1], dtype=np.int32),
            'name' : 'bottom_to_top',
            },

            {'goal_cell': np.array([3,1], dtype=np.int32),
            'reset_cell': np.array([1,1], dtype=np.int32),
            'name' : 'top_to_bottom',
            },
        ]

    elif cfg.dataset_name in ["pointmaze-medium-v0"]:
        test_start_state_goal = [
            {'goal_cell': np.array([6,3], dtype=np.int32),
            'reset_cell': np.array([6,6], dtype=np.int32),
            'name' : 'bottom_right_to_bottom_center',
            },

            {'goal_cell': np.array([6,1], dtype=np.int32),
            'reset_cell': np.array([6,6], dtype=np.int32),
            'name' : 'bottom_right_to_bottom_left',
            },

            {'goal_cell': np.array([6,3], dtype=np.int32),
            'reset_cell': np.array([6,5], dtype=np.int32),
            'name' : 'bottom_rightish_to_bottom_center',
            },

            {'goal_cell': np.array([6,1], dtype=np.int32),
            'reset_cell': np.array([6,5], dtype=np.int32),
            'name' : 'bottom_rightish_to_bottom_left',
            },

            {'goal_cell': np.array([6,5], dtype=np.int32),
            'reset_cell': np.array([1,1], dtype=np.int32),
            'name' : 'top_left_to_bottom_rightish',
            },

            {'goal_cell': np.array([6,6], dtype=np.int32),
            'reset_cell': np.array([1,6], dtype=np.int32),
            'name' : 'top_right_to_bottom_right',
            },
        ]
    
    elif cfg.dataset_name in ["antmaze-medium-v0"]:
        test_start_state_goal = [
            {'goal_cell': np.array([6,5], dtype=np.int32),
            'reset_cell': np.array([6,1], dtype=np.int32),
            'name' : 'bottom_left_to_bottom_rightish',
            },

            {'goal_cell': np.array([1,6], dtype=np.int32),
            'reset_cell': np.array([6,1], dtype=np.int32),
            'name' : 'bottom_left_to_top_right',
            },

            {'goal_cell': np.array([6,1], dtype=np.int32),
            'reset_cell': np.array([6,5], dtype=np.int32),
            'name' : 'bottom_rightish_to_bottom_left',
            },

            {'goal_cell': np.array([1,6], dtype=np.int32),
            'reset_cell': np.array([6,5], dtype=np.int32),
            'name' : 'bottom_rightish_to_top_right',
            },

            {'goal_cell': np.array([6,1], dtype=np.int32),
            'reset_cell': np.array([1,6], dtype=np.int32),
            'name' : 'top_right_to_bottom_left',
            },

             {'goal_cell': np.array([6,5], dtype=np.int32),
            'reset_cell': np.array([1,6], dtype=np.int32),
            'name' : 'top_right_to_bottom_rightish',
            },

        ]

    elif cfg.dataset_name in ["pointmaze-large-v0"]:
        test_start_state_goal = [
            {'goal_cell': np.array([7,4], dtype=np.int32),
            'reset_cell': np.array([7,1], dtype=np.int32),
            'name' : 'bottom_left_to_bottom_center',
            },

            {'goal_cell': np.array([7,10], dtype=np.int32),
            'reset_cell': np.array([7,1], dtype=np.int32),
            'name' : 'bottom_left_to_bottom_right',
            },

            {'goal_cell': np.array([1,10], dtype=np.int32),
            'reset_cell': np.array([7,1], dtype=np.int32),
            'name' : 'bottom_left_to_top_right',
            },

            {'goal_cell': np.array([7,1], dtype=np.int32),
            'reset_cell': np.array([7,4], dtype=np.int32),
            'name' : 'bottom_center_to_bottom_left',
            },

            {'goal_cell': np.array([7,10], dtype=np.int32),
            'reset_cell': np.array([7,4], dtype=np.int32),
            'name' : 'bottom_center_to_bottom_right',
            },

            {'goal_cell': np.array([1,1], dtype=np.int32),
            'reset_cell': np.array([7,4], dtype=np.int32),
            'name' : 'bottom_center_to_top_left',
            },

            {'goal_cell': np.array([7,1], dtype=np.int32),
            'reset_cell': np.array([7,10], dtype=np.int32),
            'name' : 'bottom_right_to_bottom_left',
            }
        ]

    elif cfg.dataset_name in ["antmaze-large-v0"]:
        test_start_state_goal = [
            {'goal_cell': np.array([7,4], dtype=np.int32),
            'reset_cell': np.array([7,1], dtype=np.int32),
            'name' : 'bottom_left_to_bottom_center',
            },

            {'goal_cell': np.array([7,10], dtype=np.int32),
            'reset_cell': np.array([7,1], dtype=np.int32),
            'name' : 'bottom_left_to_bottom_right',
            },

            {'goal_cell': np.array([1,10], dtype=np.int32),
            'reset_cell': np.array([7,1], dtype=np.int32),
            'name' : 'bottom_left_to_top_right',
            },

            {'goal_cell': np.array([7,1], dtype=np.int32),
            'reset_cell': np.array([7,4], dtype=np.int32),
            'name' : 'bottom_center_to_bottom_left',
            },

            {'goal_cell': np.array([1,1], dtype=np.int32),
            'reset_cell': np.array([7,4], dtype=np.int32),
            'name' : 'bottom_center_to_top_left',
            },

            {'goal_cell': np.array([7,1], dtype=np.int32),
            'reset_cell': np.array([7,10], dtype=np.int32),
            'name' : 'bottom_right_to_bottom_left',
            }
        ]

    else:
        raise NotImplementedError
    
    return test_start_state_goal

def get_maze_map(dataset_name):
    return None

def cell_to_state(cell, maze):
    return cell[:, 0] * maze.map_width + cell[:, 1]


def cell_xy_to_rowcol(maze, xy_pos: np.ndarray) -> np.ndarray:
        """Converts a cell x and y coordinates to `(i,j)`"""

        i = np.reshape(np.floor((maze.y_map_center - xy_pos[:, 1]) / maze.maze_size_scaling), (-1, 1))
        j = np.reshape(np.floor((xy_pos[:, 0] + maze.x_map_center) / maze.maze_size_scaling), (-1, 1))

        return np.concatenate([i,j], axis=-1)