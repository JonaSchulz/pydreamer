import argparse
import numpy as np
import torch
from distutils.util import strtobool
import gymnasium
import matplotlib.pyplot as plt

from pydreamer.data import MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.functions import map_structure
from pydreamer.preprocessing import Preprocessor, to_onehot
from pydreamer.tools import *
from dqn.agent import DQN


model_path = "mlruns/0/1420c2c9944e43aa9463e045ed4ac11f/artifacts/checkpoints/latest.pt"
env_id = "MinAtar/Seaquest-v0"
num_steps = 10000
dqn_path = "dqn/seaquest/model_100000.pt"


def main(worker_id=0,
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         model_conf=dict(),
         ):

    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat, worker_id)
    policy = create_policy("network", env, model_conf)
    policy_dqn = create_policy("dqn", env, model_conf, env_id)
    policy_dqn.model.load_state_dict(torch.load(dqn_path, map_location='cpu'))

    if isinstance(policy, NetworkPolicy):
        # takes ~10sec to load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint['model_state_dict'])

    features = []
    player_pos_2d = []

    obs = env.reset()
    for _ in range(num_steps):
        player_pos_2d.append(np.argwhere(obs['image'] == 1)[0])
        _, _, feats = policy(obs)
        action, _ = policy_dqn(obs)
        features.append(feats.squeeze())
        obs, reward, done, inf = env.step(action)
        if done:
            obs = env.reset()

    player_pos_x = np.array([i[1] for i in player_pos_2d])
    player_pos_y = np.array([i[0] for i in player_pos_2d])
    player_pos_flat = player_pos_y * 10 + player_pos_x

    features = np.asarray(features).T
    corr_x = np.array([np.corrcoef(player_pos_x, i)[0, 1] for i in features])
    corr_y = np.array([np.corrcoef(player_pos_y, i)[0, 1] for i in features])
    corr_flat = np.array([np.corrcoef(player_pos_flat, i)[0, 1] for i in features])

    corr_flat[np.isnan(corr_flat)] = 0
    max_corr_index_flat = np.argmax(np.abs(corr_flat))
    max_corr_features_flat = features[max_corr_index_flat, :]

    features_count = np.zeros(100, dtype=float)
    for pos in range(100):
        indices = player_pos_flat == pos
        if not indices.sum():
            continue
        features_count[pos] = max_corr_features_flat[indices].mean()
    features_count = features_count.reshape((10, 10))

    print(max_corr_index_flat)
    print(corr_flat[max_corr_index_flat])

    plt.imshow(features_count)
    plt.colorbar()
    plt.show()


def create_policy(policy_type: str, env, model_conf, env_id=None):
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


class NetworkPolicy:
    def __init__(self, model: Dreamer, preprocess: Preprocessor):
        self.model = model
        self.preprocess = preprocess
        self.state = model.init_state(1)

    def __call__(self, obs) -> Tuple[np.ndarray, dict]:
        batch = self.preprocess.apply(obs, expandTB=True)
        obs_model: Dict[str, Tensor] = map_structure(batch, torch.from_numpy)  # type: ignore

        with torch.no_grad():
            features, out_state = self.model.wm.forward(obs_model, self.state)
            action_distr, new_state, metrics = self.model.inference(obs_model, self.state)
            action = action_distr.sample()
            self.state = new_state

        metrics = {k: v.item() for k, v in metrics.items()}
        metrics.update(action_prob=action_distr.log_prob(action).exp().mean().item(),
                       policy_entropy=action_distr.entropy().mean().item())

        action = action.squeeze()  # (1,1,A) => A
        return action.numpy(), metrics, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_steps', type=int, default='100')
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

    main(env_action_repeat=conf.env_action_repeat, model_conf=conf)
