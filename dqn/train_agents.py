from agent import Agent
import os
import torch

env_ids = ['MinAtar/Asterix-v0', 'MinAtar/Breakout-v0', 'MinAtar/Seaquest-v0', 'MinAtar/Freeway-v0']
model_dirs = ['asterix', 'breakout', 'seaquest', 'freeway']
num_steps = [100, 1000, 10000, 1000000, 10000000]


for i, _id in enumerate(env_ids):
    agent = Agent(_id)
    for num_step in num_steps:
        agent.train(num_step, save_interval=None, save_path=os.path.join(model_dirs[i], f'model_{num_step}.pt'))
