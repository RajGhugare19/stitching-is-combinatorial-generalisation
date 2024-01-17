import hydra
import wandb
import random
import minari
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DecisionMLP
from utils import MinariEpisodicDataset, convert_remote_to_local, get_test_start_state_goals, get_lr, AntmazeWrapper 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
         
def eval_env(cfg, model, device, render=False):
    if render:
        render_mode = 'human'
    else:
        render_mode = None

    if "pointmaze" in cfg.dataset_name:
        env = env = gym.make(cfg.env_name, continuing_task=False, render_mode=render_mode)
    elif "antmaze" in cfg.dataset_name:
        env = AntmazeWrapper(env = gym.make(cfg.env_name, continuing_task=False, render_mode=render_mode))
    else:
        raise NotImplementedError    

    test_start_state_goal = get_test_start_state_goals(cfg)
    
    model.eval()
    results = dict()
    with torch.no_grad():
        cum_reward = 0
        for ss_g in test_start_state_goal:
            total_reward = 0
            total_timesteps = 0
            
            print(ss_g['name'] + ':')
            for _ in range(cfg.num_eval_ep):
                obs, _ = env.reset(options=ss_g)
                done = False
                for t in range(env.spec.max_episode_steps):
                    total_timesteps += 1

                    running_state = torch.tensor(obs['observation'], dtype=torch.float32, device=device).view(1, -1)
                    target_goal = torch.tensor(obs['desired_goal'], dtype=torch.float32, device=device).view(1, -1)
                    
                    act_preds = model.forward(
                            running_state,
                            target_goal,
                            )
                    act = act_preds[0].detach()

                    obs, running_reward, done, _, _ = env.step(act.cpu().numpy())

                    total_reward += running_reward

                    if done:
                        break
                
                print('Achievied goal: ', tuple(obs['achieved_goal'].tolist()))
                print('Desired goal: ', tuple(obs['desired_goal'].tolist()))
            
            print("=" * 60)
            cum_reward += total_reward
            results['eval/' + str(ss_g['name']) + '_avg_reward'] = total_reward / cfg.num_eval_ep
            results['eval/' + str(ss_g['name']) + '_avg_ep_len'] = total_timesteps / cfg.num_eval_ep
        
        results['eval/avg_reward'] = cum_reward / (cfg.num_eval_ep * len(test_start_state_goal))
        env.close()
    return results

def train(cfg, hydra_cfg):

    #set seed
    set_seed(cfg.seed)

    #set device
    device = torch.device(cfg.device)

    if cfg.save_snapshot:
        checkpoint_path = Path(hydra_cfg['runtime']['output_dir']) / Path('checkpoint')
        checkpoint_path.mkdir(exist_ok=True)
        best_eval_returns = 0

    start_time = datetime.now().replace(microsecond=0)
    time_elapsed = start_time - start_time
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    if "pointmaze" in cfg.dataset_name:
        if "umaze" in cfg.dataset_name:
            cfg.env_name = 'PointMaze_UMaze-v3'
            cfg.nclusters = 20 if cfg.nclusters is None else cfg.nclusters
        elif "medium" in cfg.dataset_name:
            cfg.env_name = 'PointMaze_Medium-v3'
            cfg.nclusters = 40 if cfg.nclusters is None else cfg.nclusters
        elif "large" in cfg.dataset_name:
            cfg.env_name = 'PointMaze_Large-v3'
            cfg.nclusters = 80 if cfg.nclusters is None else cfg.nclusters
        env = gym.make(cfg.env_name, continuing_task=False)

    elif "antmaze" in cfg.dataset_name:
        if "umaze" in cfg.dataset_name:
            cfg.env_name = 'AntMaze_UMaze-v4'
            cfg.nclusters = 20 if cfg.nclusters is None else cfg.nclusters
        elif "medium" in cfg.dataset_name:
            cfg.env_name = 'AntMaze_Medium-v4'
            cfg.nclusters = 40 if cfg.nclusters is None else cfg.nclusters
        elif "large" in cfg.dataset_name:
            cfg.env_name = 'AntMaze_Large-v4'
            cfg.nclusters = 80 if cfg.nclusters is None else cfg.nclusters
        else:
            raise NotImplementedError
        env = AntmazeWrapper(gym.make(cfg.env_name, continuing_task=False))

    else:
        raise NotImplementedError
    
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)

    #create dataset
    if cfg.remote_data:
        convert_remote_to_local(cfg.dataset_name, env)
   
    train_dataset = MinariEpisodicDataset(cfg.dataset_name, cfg.remote_data, cfg.augment_data, cfg.augment_prob, cfg.nclusters)
    
    train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=cfg.num_workers
                        )
    train_data_iter = iter(train_data_loader)

    #create model
    model = DecisionMLP(cfg.env_name, env, goal_dim=train_dataset.goal_dim).to(device)

    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=cfg.lr,
                    )

    total_updates = 0
    for i_train_iter in range(cfg.max_train_iters):

        log_action_losses = []
        model.train()
        
        for i in range(cfg.num_updates_per_iter):            
            try:
                states, goals, actions = next(train_data_iter)

            except StopIteration:
                train_data_iter = iter(train_data_loader)
                states, goals, actions = next(train_data_iter)

            states = states.to(device)         
            actions = actions.to(device)                        
            goals = goals.to(device)                            

            action_preds = model.forward(
                                states=states, 
                                goals=goals,
                            )
            action_loss = F.mse_loss(action_preds, actions, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            log_action_losses.append(action_loss.detach().cpu().item())

        time = datetime.now().replace(microsecond=0) - start_time - time_elapsed
        time_elapsed = datetime.now().replace(microsecond=0) - start_time

        total_updates += cfg.num_updates_per_iter
        
        mean_action_loss = np.mean(log_action_losses)

        results = eval_env(cfg, model, device, render=cfg.render)

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + str(time_elapsed)  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "train action loss: " +  format(mean_action_loss, ".5f") #+ '\n' +
            )
        
        print(results)
        print(log_str)
        
        if cfg.wandb_log:
            log_data = dict()
            log_data['time'] =  time.total_seconds()
            log_data['time_elapsed'] =  time_elapsed.total_seconds()
            log_data['total_updates'] =  total_updates
            log_data['mean_action_loss'] =  mean_action_loss
            log_data['lr'] = get_lr(optimizer)
            log_data['training_iter'] = i_train_iter
            log_data.update(results)
            wandb.log(log_data)

        if cfg.save_snapshot and (1+i_train_iter)%cfg.save_snapshot_interval == 0:
            snapshot = Path(checkpoint_path) / Path(str(i_train_iter)+'.pt')
            torch.save(model.state_dict(), snapshot)

        if cfg.save_snapshot and results['eval/avg_reward'] >= best_eval_returns:
            print("=" * 60)
            print("saving best model!")
            print("=" * 60)
            best_eval_returns = results['eval/avg_reward']
            snapshot = Path(checkpoint_path) / 'best.pt'
            torch.save(model.state_dict(), snapshot)

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("=" * 60)
    
@hydra.main(config_path='cfgs', config_name='dmlp', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        if cfg.wandb_dir is None:
            cfg.wandb_dir = hydra_cfg['runtime']['output_dir']

        project_name = cfg.dataset_name
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=cfg.wandb_dir, group=cfg.wandb_group_name)
        wandb.run.name = cfg.wandb_run_name
    
    train(cfg, hydra_cfg)
        
if __name__ == "__main__":
    main()