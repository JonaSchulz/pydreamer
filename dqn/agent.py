import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import math
from itertools import count
from multiprocessing import Process

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)

        return x


class Agent:
    # Hyperparameters:
    BATCH_SIZE = 128
    MEMORY = 10000
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    def __init__(self, env_id):
        self.env_id = env_id
        self.env = gym.make(env_id)

        n_actions = self.env.action_space.n
        state, info = self.env.reset()
        n_observations = state.size

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Agent.LR, amsgrad=True)
        self.memory = ReplayMemory(Agent.MEMORY)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * \
            math.exp(-1. * self.steps_done / Agent.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < Agent.BATCH_SIZE:
            return
        transitions = self.memory.sample(Agent.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(Agent.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * Agent.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_steps, save_interval=None, save_path=None):
        steps = 0
        for i_episode in count():
            # Initialize the environment and get it's state
            if steps >= num_steps:
                break
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            for t in count():
                steps += 1
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*Agent.TAU + target_net_state_dict[key]*(1-Agent.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if steps >= num_steps:
                    break

                if done:
                    #plot_durations()
                    break

            if save_interval is not None and (i_episode % save_interval == 0) and i_episode:
                assert save_path is not None, 'Specify model save path'
                torch.save(self.target_net.state_dict(), save_path)

        if save_path is not None:
            torch.save(self.target_net.state_dict(), save_path)

    def play(self, num_episodes):
        env = gym.make(self.env_id, render_mode="human")
        observation, info = env.reset(seed=42)
        state = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)
        episodes = 0

        while episodes < num_episodes:
            action = self.policy_net(state).max(1)[1].view(1, 1).item()
            observation, reward, terminated, truncated, info = env.step(action)
            state = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)

            if terminated or truncated:
                observation, info = env.reset()
                episodes += 1

        self.env.close()


def run(conf):
    assert conf.train_dqn and conf.dqn_train_steps, 'Set train_dqn True and set dqn_train_steps'
    num_steps = [conf.dqn_train_steps for _ in conf.env_id] if type(conf.dqn_train_steps) is int else conf.dqn_train_steps
    save_interval = [conf.dqn_save_interval for _ in conf.env_id] if type(conf.dqn_save_interval) is int else conf.dqn_save_interval
    agents = [Agent(_id) for _id in conf.env_id]





