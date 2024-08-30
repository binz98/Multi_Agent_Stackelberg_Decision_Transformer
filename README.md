# Stackelberg Decision  Transformer

This is the implementation of Stackelberg Decision  Transformer (STEER). 

## Installation

#### Dependences

```
pip install -r requirements.txt
```

### Multi-agent MuJoCo

Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:

```
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### Google Research Football

Please following the instructios in https://github.com/google-research/football.

## How to run
When your environment is ready, you could run shells in the "scripts" folder with algo="steer". For example:
``` Bash
./train_matrix.sh
./train_football.sh
./train_mujoco.sh
```
If you would like to change the configs of experiments, you could modify sh files or look for config.py for more details.



