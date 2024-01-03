import argparse
import numpy as np
import torch
from distutils.util import strtobool
import gymnasium

from pydreamer.data import MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.functions import map_structure
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import *
from generator import create_policy, RandomPolicy, NetworkPolicy

model_path = "mlruns/0/1420c2c9944e43aa9463e045ed4ac11f/artifacts/checkpoints/latest.pt"


def main(model_path=None,
         num_episodes=50,
         env_id='MinAtar/Breakout-v0',
         worker_id=0,
         policy='random',
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         model_conf=dict(),
         ):

    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat, worker_id,
                     render_mode="human")
    policy = create_policy(policy, env, model_conf)

    if isinstance(policy, NetworkPolicy):
        # takes ~10sec to load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint['model_state_dict'])

    for _ in range(num_episodes):
        epsteps = 0
        obs = env.reset()
        done = False

        while not done:
            action, mets = policy(obs)
            obs, reward, done, inf = env.step(action)
            epsteps += 1

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--policy', type=str, default='random')
    parser.add_argument('--env_id', type=str, default='MinAtar/Breakout-v0')
    parser.add_argument('--num_episodes', type=int, default='50')
    args, remaining = parser.parse_known_args()

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    main(model_path=args.model_path if args.model_path is not None else model_path,
         num_episodes=args.num_episodes,
         env_id=args.env_id,
         policy=args.policy,
         env_action_repeat=conf.env_action_repeat,
         model_conf=conf
         )
