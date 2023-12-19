import gymnasium
import numpy as np


class CategoricalWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs_index = np.argmax(obs, -1) + 1
        obs_index[obs.sum(-1) == 0] = 0
        return obs_index


class DictWrapperGymnasium(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs):
        if isinstance(obs, dict):
            return obs  # Already a dictionary
        if len(obs.shape) == 1:
            return {'vecobs': obs.astype(float)}  # Vector env
        else:
            return {'image': obs.astype(int)}  # Image env


class ActionRewardResetWrapperGymnasium(gymnasium.Wrapper):

    def __init__(self, env, no_terminal):
        super().__init__(env)
        self.env = env
        self.no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self.action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        action = int(action)

        if isinstance(action, int):
            action_vec = np.zeros(self.action_size)
            action_vec[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (self.action_size,), "Wrong one-hot action shape"
            action_vec = action
        obs['action'] = action_vec
        obs['reward'] = np.array(reward)
        obs['terminal'] = np.array(False if self.no_terminal or truncated else done)
        obs['reset'] = np.array(False)
        return obs, reward, done, info

    def reset(self):
        obs, info = self.env.reset()
        obs['action'] = np.zeros(self.action_size)
        obs['reward'] = np.array(0.0)
        obs['terminal'] = np.array(False)
        obs['reset'] = np.array(True)
        return obs


class OneHotActionWrapperGymnasium(gymnasium.Wrapper):
    """Allow to use one-hot action on a discrete action environment."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Note: we don't want to change env.action_space to Box(0., 1., (n,)) here,
        # because then e.g. RandomPolicy starts generating continuous actions.

    def step(self, action):
        if not isinstance(action, np.int64):
            action = action.argmax()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()