# Adapting to the Unfamiliar
This project is built on top of the pydreamer repository, a PyTorch implementation of DreamerV2, found at: https://github.com/jurgisp/pydreamer.
A variety of new scripts have been added to perform the experiments of our project.

## Our Code
- generator_multi_env.py: This script allows the generation of a multi-environment dataset.
The dataset is stored in the form of an mlflow artifacts directory containing files of episodes collected from the different environments. DQN agents can be used as a policy for action sampling.
- train_wm_only.py: This script trains a DreamerV2 world model on an existing multi-environment dataset.
- train_finetune.py: This script fine-tunes a multi-environment world model on a new environment, training a complete DreamerV2 agent in the process. World model and actor can be trained for different numbers of steps.
- dqn: The dqn package contains code for training and storing MinAtar DQN agents that can subsequently be used for multi-environment dataset generation.
- atari_net: The atari_net package contains the code for the model architecture of DQN agents taken from https://github.com/floringogianu/atari-agents. Download checkpoint files from their "DQN_modern" category.

## Running
### Training a DreamerV2 agent with the original training procedure:
- on an Atari environment (replace Pong with the desired game):
```sh
python launch.py --configs defaults atari --env_id Atari-Pong
```
- on a MinAtar environment (replace Breakout with the desired game):
```sh
python launch.py --configs defaults minatar --env_id MinAtar/Breakout-v0
```

### Training MinAtar DQN agents:
```sh
python dqn/train_agents.py
```
Which environments to train on, how many checkpoints to create per environment and where to store checkpoint files can be adjusted in the script itself.

### Generating a multi-environment dataset:
Put the desired games in defaults.yaml as the `env_id` argument (for MinAtar in the `generator_multi_env` subsection, for Atari in the `generator_multi_env_atari` subsection).
- for Atari games run:
```sh
python launch_generator_multi_env.py --configs defaults atari generator_multi_env_atari
```
- for MinAtar games run:
```sh
python launch_generator_multi_env.py --configs defaults atari generator_multi_env
```

### Training a DreamerV2 multi-environment world model:
Requires a finished generator_multi_env run with mlflow id `run_id`. Then run:
```sh
python launch_wm_training.py --configs defaults minatar wm_training --mlflow_run_id run_id
```
Replace `minatar` with `atari` in the case of Atari game environments.

### Finetuning a multi-environment world model on a new game environment:
Requires a trained multi-environment world model with model checkpoint file at `model_path`.
- on a MinAtar environment (replace SpaceInvaders with the desired game):
```sh
python launch_finetuning.py --configs defaults minatar finetune --model_checkpoint model_path --env_id MinAtar/SpaceInvaders-v0
```
- on an Atari environment (replace Pong with the desired game):
```sh
python launch_finetuning.py --configs defaults atari finetune --model_checkpoint model_path --env_id Atari-Pong
```
To adjust the number of training steps for world model and actor as well as to set whether to train the RSSM or not, change the corresponding parameters in defaults.yaml in the `finetune` subsection.