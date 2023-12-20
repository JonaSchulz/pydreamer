from agent import Agent

agent = Agent('MinAtar/Seaquest-v0')
agent.train(100, save_interval=100, save_path="model_seaquest.pt")
