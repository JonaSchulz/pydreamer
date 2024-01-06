import argparse
import logging
import logging.config
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from logging import critical, debug, error, info, warning
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import cv2

import mlflow
import numpy as np
import torch
import torch.distributions as D

from pydreamer.data import MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.functions import map_structure
from pydreamer.preprocessing import Preprocessor, to_onehot
from pydreamer.tools import *
from dqn.agent import DQN
from atari_net.model import AtariNet, _load_checkpoint


def load_dqn_model(model, model_dir):
    model_file = os.path.join(model_dir, random.choice(os.listdir(model_dir)))
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    return model_file


def load_atari_net_model(policy, model_dir):
    model_file = os.path.join(model_dir, random.choice(os.listdir(model_dir)))
    policy.load_checkpoint(model_file)
    return model_file


def main(env_id=['MiniGrid-MazeS11N-v0'],
         save_uri=None,
         save_uri2=None,
         worker_id=0,
         policy_main='random',
         policy_prefill='random',
         num_steps=int(1e6),
         num_steps_prefill=0,
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         limit_step_ratio=0.0,
         steps_per_npz=1000,
         model_reload_interval=120,
         model_conf=dict(),
         log_mlflow_metrics=True,
         split_fraction=0.0,
         metrics_prefix='agent',
         metrics_gamma=0.99,
         log_every=10
         ):

    configure_logging(prefix=f'[GEN {worker_id}]', info_color=LogColorFormatter.GREEN)
    mlrun = mlflow_init()
    info(f'Generator {worker_id} started:'
         f' env={[_id for _id in env_id]}'
         f', n_steps={num_steps:,}'
         f', n_prefill={num_steps_prefill:,}'
         f', split_fraction={split_fraction}'
         f', metrics={metrics_prefix if log_mlflow_metrics else None}'
         f', save_uri={save_uri}')

    if not save_uri:
        save_uri = f'{mlrun.info.artifact_uri}/episodes/{worker_id}'
    if not save_uri2:
        assert split_fraction == 0.0, 'Specify two save destinations, if splitting'

    repository = MlflowEpisodeRepository(save_uri)
    repository2 = MlflowEpisodeRepository(save_uri2) if save_uri2 else repository
    nfiles, steps_saved, episodes = repository.count_steps()
    info(f'Found existing {nfiles} files, {episodes} episodes, {steps_saved} steps in {repository}')

    # Env
    # if env_id == "minatar":
    #    env_id = ['MinAtar/Asterix-v0', 'MinAtar/Breakout-v0', 'MinAtar/Freeway-v0', 'MinAtar/Seaquest-v0']
    env = [create_env(_id, env_no_terminal, env_time_limit, env_action_repeat, worker_id) for _id in env_id]

    # Policy

    if num_steps_prefill:
        # Start with prefill policy
        info(f'Prefill policy: {policy_prefill}')
        policy = [create_policy(policy_prefill, _env, model_conf, id) for id, _env in zip(env_id, env)]
        is_prefill_policy = True
    else:
        info(f'Policy: {policy_main}')
        policy = [create_policy(policy_main, _env, model_conf, id) for id, _env in zip(env_id, env)]
        is_prefill_policy = False

    # RUN

    datas = []
    last_model_load = 0
    model_step = 0
    metrics_agg = defaultdict(list)
    all_returns = []
    steps = 0
    steps_per_env = np.array([0 for _env in env])

    while steps_saved < num_steps:

        # Switch policy prefill => main

        if is_prefill_policy and steps_saved >= num_steps_prefill:
            info(f'Switching to main policy: {policy_main}')
            policy = [create_policy(policy_main, _env, model_conf, id) for id, _env in zip(env_id, env)]
            is_prefill_policy = False

        # Load network

        if time.time() - last_model_load > model_reload_interval:
            for i, _policy in enumerate(policy):
                if isinstance(_policy, AtariNetPolicy):
                    model_file = load_atari_net_model(_policy, model_conf.atari_net_dir[env_id[i]])
                    info(f"Loading model checkpoint {model_file}")

                elif isinstance(_policy, DQNPolicy):
                    model_file = load_dqn_model(_policy.model, model_conf.dqn_dir[env_id[i]])
                    info(f"Loading model checkpoint {model_file}")

                elif isinstance(_policy, NetworkPolicy):
                    while True:
                        # takes ~10sec to load checkpoint
                        model_step = mlflow_load_checkpoint(_policy.model, artifact_path=conf.model_path[i], map_location='cpu')  # type: ignore
                        if model_step:
                            info(f'Generator loaded model checkpoint {model_step}')
                            break
                        else:
                            debug('Generator model checkpoint not found, waiting...')
                            time.sleep(10)
            last_model_load = time.time()

            if limit_step_ratio and steps_saved >= model_step * limit_step_ratio:
                # Rate limiting - keep looping until new model checkpoint is loaded
                time.sleep(1)
                continue

        # Unroll one episode
        # env_index = np.random.randint(0, len(env))
        env_index = np.argmin(steps_per_env)
        info(f"Environment: {env_id[env_index]}")
        # print(f"Steps per env: {list(zip(env_id, steps_per_env))}")
        epsteps = 0
        timer = time.time()
        obs = env[env_index].reset()
        done = False
        metrics = defaultdict(list)

        while not done:
            action, mets = policy[env_index](obs)
            obs, reward, done, inf = env[env_index].step(action)
            steps += 1
            epsteps += 1
            for k, v in mets.items():
                metrics[k].append(v)
            steps_per_env[env_index] += 1

        episodes += 1
        data = inf['episode'] # type: ignore
        data['env_id'] = env_index * np.ones_like(data['terminal'], dtype=int)
        if 'policy_value' in metrics:
            data['policy_value'] = np.array(metrics['policy_value'] + [np.nan])     # last terminal value is null
            data['policy_entropy'] = np.array(metrics['policy_entropy'] + [np.nan])  # last policy is null
            data['action_prob'] = np.array([np.nan] + metrics['action_prob'])       # first action is null
        else:
            # Need to fill with placeholders, so all batches have the same keys
            data['policy_value'] = np.full(data['reward'].shape, np.nan)
            data['policy_entropy'] = np.full(data['reward'].shape, np.nan)
            data['action_prob'] = np.full(data['reward'].shape, np.nan)

        # Log

        fps = epsteps / (time.time() - timer + 1e-6)
        print_once('Episode data sample: ', {k: v.shape for k, v in data.items()})

        info(f"Episode recorded:"
             f"  steps: {epsteps}"
             f",  reward: {data['reward'].sum():.1f}"
             f",  terminal: {data['terminal'].sum():.0f}"
            #  f",  visited: {(data.get('map_seen', np.zeros(1))[-1] > 0).mean():.1%}"
             f",  total steps: {steps:.0f}"
             f",  episodes: {episodes}"
             f",  saved steps (train): {steps_saved:.0f}"
             f",  fps: {fps:.0f}"
             )

        if log_mlflow_metrics:
            metrics = {f'{metrics_prefix}/{k}': np.array(v).mean() for k, v in metrics.items()}
            all_returns.append(data['reward'].sum())
            metrics.update({
                f'{metrics_prefix}/episode_length': epsteps,
                f'{metrics_prefix}/fps': fps,
                f'{metrics_prefix}/steps': steps,  # All steps since previous restart
                f'{metrics_prefix}/steps_saved': steps_saved,  # Steps saved in the training repo
                f'{metrics_prefix}/env_steps': steps * env_action_repeat,
                f'{metrics_prefix}/episodes': episodes,
                f'{metrics_prefix}/return': data['reward'].sum(),
                f'{metrics_prefix}/return_cum': np.array(all_returns[-100:]).mean(),
            })

            # Calculate return_discounted

            rewards_v = data['reward'].copy()
            if not data['terminal'][-1]:
                avg_value = rewards_v.mean() / (1.0 - metrics_gamma)
                rewards_v[-1] += avg_value
            returns_discounted = discount(rewards_v, gamma=metrics_gamma)
            metrics[f'{metrics_prefix}/return_discounted'] = returns_discounted.mean()

            # Calculate policy_value_terminal

            if data['terminal'][-1]:
                value_terminal = data['policy_value'][-2] - data['reward'][-1]  # This should be zero, because value[last] = reward[last]
                metrics[f'{metrics_prefix}/policy_value_terminal'] = value_terminal

            # Goal visibility metrics for Scavenger

            if 'goals_visage' in data:
                goals_seen = data['goals_visage'] < 1e5
                metrics[f'{metrics_prefix}/goals_seen_avg'] = goals_seen.sum(axis=-1).mean()
                metrics[f'{metrics_prefix}/goals_seen_last'] = goals_seen[-1].sum()
                metrics[f'{metrics_prefix}/goals_seenage'] = (data['goals_visage'] * goals_seen).sum() / goals_seen.sum()

            # Aggregate every 10 episodes

            for k, v in metrics.items():
                if not np.isnan(v):
                    metrics_agg[k].append(v)

            if len(metrics_agg[f'{metrics_prefix}/return']) >= log_every:
                metrics_agg_max = {k: np.array(v).max() for k, v in metrics_agg.items()}
                metrics_agg = {k: np.array(v).mean() for k, v in metrics_agg.items()}
                metrics_agg[f'{metrics_prefix}/return_max'] = metrics_agg_max[f'{metrics_prefix}/return']
                metrics_agg['_timestamp'] = datetime.now().timestamp()
                mlflow_log_metrics(metrics_agg, step=model_step)
                metrics_agg = defaultdict(list)

        # Save to npz

        datas.append(data)
        datas_episodes = len(datas)
        datas_steps = sum(len(d['reset']) - 1 for d in datas)

        if datas_steps >= steps_per_npz:

            # Concatenate episodes

            data = {}
            for key in datas[0]:
                data[key] = np.concatenate([b[key] for b in datas], axis=0)
            datas = []
            print_once('Collected data sample: ', {k: v.shape for k, v in data.items()})

            # ... or chunk

            # if steps_per_npz=1000, then chunk size will be [1000,1999]
            if datas_steps >= 2 * steps_per_npz:
                chunks = chunk_episode_data(data, steps_per_npz)
            else:
                chunks = [data]

            # Save to npz

            repo = repository if (np.random.rand() > split_fraction) else repository2
            for i, data in enumerate(chunks):
                if 'image' in data and len(data['image'].shape) == 4:
                    # THWC => HWCT for better compression
                    data['image_t'] = data['image'].transpose(1, 2, 3, 0)
                    del data['image']
                else:
                    # Categorical image, leave it alone
                    pass
                repo.save_data(data, episodes - datas_episodes, episodes - 1, i)

            if repo == repository:
                # Only count steps in the training repo, so that prefill and limit_step_ratio works correctly
                steps_saved += datas_steps

    info('Generator done.')


def create_policy(policy_type: str, env, model_conf, env_id=None):
    if policy_type == 'atari_net':
        assert env_id is not None, 'Specify env_id'
        return AtariNetPolicy(env_id)

    if policy_type == 'dqn':
        assert env_id is not None, 'Specify env_id'
        return DQNPolicy(env_id, model_conf.image_size, model_conf.action_dim)

    if policy_type == 'network':
        conf = model_conf
        if conf.model == 'dreamer':
            model = Dreamer(conf)
        else:
            assert False, conf.model
        preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                                  image_key=conf.image_key,
                                  map_categorical=conf.map_channels if conf.map_categorical else None,
                                  map_key=conf.map_key,
                                  action_dim=env.action_size,  # type: ignore
                                  clip_rewards=conf.clip_rewards)
        return NetworkPolicy(model, preprocess)

    if policy_type == 'random':
        return RandomPolicy(env.action_space)

    if policy_type == 'minigrid_wander':
        from pydreamer.envs.minigrid import MinigridWanderPolicy
        return MinigridWanderPolicy()

    if policy_type == 'maze_bouncing_ball':
        from pydreamer.envs.miniworld import MazeBouncingBallPolicy
        return MazeBouncingBallPolicy()

    if policy_type == 'maze_dijkstra':
        from pydreamer.envs.miniworld import MazeDijkstraPolicy
        step_size = env.params.params['forward_step'].default / env.room_size  # type: ignore
        turn_size = env.params.params['turn_step'].default  # type: ignore
        return MazeDijkstraPolicy(step_size, turn_size)

    if policy_type == 'goal_dijkstra':
        from pydreamer.envs.miniworld import MazeDijkstraPolicy
        step_size = env.params.params['forward_step'].default / env.room_size  # type: ignore
        turn_size = env.params.params['turn_step'].default  # type: ignore
        return MazeDijkstraPolicy(step_size, turn_size, goal_strategy='goal_direction', random_prob=0)

    raise ValueError(policy_type)


class AtariNetPolicy:

    action_space = {
        "Atari-Adventure": None,
        "Atari-Air_Raid": [0, 1, 3, 4, 11, 12],
        "Atari-Alien": None,
        "Atari-Amidar": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
        "Atari-Assault": [0, 1, 2, 3, 4, 11, 12],
        "Atari-Asterix": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Asteroids": [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        "Atari-Atlantis": [0, 1, 11, 12],
        "Atari-Bank_Heist": None,
        "Atari-Battle_Zone": None,
        "Atari-Beam_Rider": [0, 1, 2, 3, 4, 6, 7, 11, 12],
        "Atari-Berzerk": None,
        "Atari-Bowling": [0, 1, 2, 5, 10, 13],
        "Atari-Boxing": None,
        "Atari-Breakout": [0, 1, 3, 4],
        "Atari-Carnival": [0, 1, 3, 4, 11, 12],
        "Atari-Centipede": None,
        "Atari-Chopper_Command": None,
        "Atari-Crazy_Climber": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Defender": None,
        "Atari-Demon_Attack": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Double_Dunk": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Elevator_Action": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Enduro": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Fishing_Derby": None,
        "Atari-Freeway": None,
        "Atari-Frostbite": None,
        "Atari-Gopher": None,
        "Atari-Gravitar": None,
        "Atari-Hero": None,
        "Atari-Ice_Hockey": None,
        "Atari-Jamesbond": None,
        "Atari-Journey_Escape": [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
        "Atari-Kangaroo": None,
        "Atari-Krull": None,
        "Atari-Kung_Fu_Master": [0, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 16, 17],
        "Atari-Montezuma_Revenge": None,
        "Atari-Ms_Pacman": [0, 2, 3, 4, 5, 6, 7, 8, 9],
        "Atari-Name_This_Game": [0, 1, 3, 4, 11, 12],
        "Atari-Phoenix": [0, 1, 3, 4, 5, 11, 12, 13],
        "Atari-Pitfall": None,
        "Atari-Pong": [0, 1, 3, 4, 11, 12],
        "Atari-Pooyan": [0, 1, 2, 5, 10, 13],
        "Atari-Private_Eye": None,
        "Atari-Qbert": [0, 1, 2, 3, 4, 5],
        "Atari-Riverraid": None,
        "Atari-Road_Runner": None,
        "Atari-Robotank": None,
        "Atari-Seaquest": None,
        "Atari-Skiing": [0, 3, 4],
        "Atari-Solaris": None,
        "Atari-Space_Invaders": [0, 1, 3, 4, 11, 12],
        "Atari-Star_Gunner": [0, 1, 2, 3, 4, 5],
        "Atari-Tennis": None,
        "Atari-Time_Pilot": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
        "Atari-Tutankham": [0, 2, 3, 4, 5, 10, 11, 12],
        "Atari-Up_N_Down": [0, 1, 2, 5, 10, 13],
        "Atari-Venture": None,
        "Atari-Video_Pinball": [0, 1, 2, 3, 4, 5, 10, 11, 12],
        "Atari-Wizard_Of_Wor": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
        "Atari-Zaxxon": None
    }

    def __init__(self, env_id):
        self.env_id = env_id
        action_space = AtariNetPolicy.action_space[env_id]
        num_actions = len(action_space) if action_space is not None else 18
        self.model = AtariNet(num_actions)
        self.image_buffer = torch.zeros(4, 84, 84, dtype=torch.uint8)

    def __call__(self, obs) -> Tuple[int, dict]:
        img = self.rgb2gray(obs['image'])
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
        img = torch.from_numpy(img).to(torch.uint8)
        self.push_to_buffer(img)
        action = self.epsilon_greedy()
        return action, {}

    def push_to_buffer(self, img):
        self.image_buffer = torch.cat((self.image_buffer[1:, :, :], torch.Tensor(img.unsqueeze(dim=0))))

    def epsilon_greedy(self, eps=0.001):
        if torch.rand((1,)).item() < eps:
            return torch.randint(self.model.action_no, (1,)).item()
        q_val, argmax_a = self.model(self.image_buffer.unsqueeze(dim=0)).max(1)
        action = argmax_a.item()
        action_space = AtariNetPolicy.action_space[self.env_id]
        if action_space is not None:
            action = action_space[action]
        return action

    def load_checkpoint(self, path):
        ckpt = _load_checkpoint(path)
        try:
            self.model.load_state_dict(ckpt["estimator_state"])
        except RuntimeError as err:
            print(path)
            print(err)

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class DQNPolicy:
    categories = {'MinAtar/Asterix-v0': 5,
                  'MinAtar/Breakout-v0': 5,
                  'MinAtar/Freeway-v0': 8,
                  'MinAtar/Seaquest-v0': 11}

    def __init__(self, env_id, image_size, n_actions):
        self.categoricals = DQNPolicy.categories[env_id]
        self.model = DQN((self.categoricals - 1) * (image_size ** 2), n_actions)

    def __call__(self, obs) -> Tuple[int, dict]:
        obs['image'] = to_onehot(obs['image'], self.categoricals)[:, :, 1:]
        return self.model(torch.from_numpy(obs['image']).flatten()), {}


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}


class NetworkPolicy:
    def __init__(self, model: Dreamer, preprocess: Preprocessor):
        self.model = model
        self.preprocess = preprocess
        self.state = model.init_state(1)

    def __call__(self, obs) -> Tuple[np.ndarray, dict]:
        batch = self.preprocess.apply(obs, expandTB=True)
        obs_model: Dict[str, Tensor] = map_structure(batch, torch.from_numpy)  # type: ignore

        with torch.no_grad():
            action_distr, new_state, metrics = self.model.inference(obs_model, self.state)
            action = action_distr.sample()
            self.state = new_state

        metrics = {k: v.item() for k, v in metrics.items()}
        metrics.update(action_prob=action_distr.log_prob(action).exp().mean().item(),
                       policy_entropy=action_distr.entropy().mean().item())

        action = action.squeeze()  # (1,1,A) => A
        return action.numpy(), metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--policy_main', type=str, required=True)
    parser.add_argument('--save_uri', type=str, default='')
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--env_time_limit', type=int, default=0)
    parser.add_argument('--env_action_repeat', type=int, default=1)
    parser.add_argument('--steps_per_npz', type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))
