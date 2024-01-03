from agent import Agent
import os
import torch

env_ids = ['MinAtar/Asterix-v0', 'MinAtar/Breakout-v0', 'MinAtar/Seaquest-v0', 'MinAtar/Freeway-v0']
model_dirs = ['asterix', 'breakout', 'seaquest', 'freeway']
num_steps = [1000000, 10000000]

# agent = Agent(env_ids[3])
# agent.policy_net.load_state_dict(torch.load("freeway/model_100000.pt", map_location="cpu"))
# agent.play(50)
# exit()

for i, _id in enumerate(env_ids):
    agent = Agent(_id)
    for num_step in num_steps:
        agent.train(num_step, save_interval=None, save_path=os.path.join(model_dirs[i], f'model_{num_step}.pt'))
