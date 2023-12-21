from agent import Agent
import os

env_ids = ['MinAtar/Asterix-v0', 'MinAtar/Breakout-v0', 'MinAtar/Seaquest-v0', 'MinAtar/Freeway-v0']
model_dirs = ['asterix', 'breakout', 'seaquest', 'freeway']

for i, _id in enumerate(env_ids):
    agent = Agent(_id)
    agent.train(4000, save_interval=None, save_path=os.path.join(model_dirs[i], 'model.pt'))
