# Stackelberg Decision  Transformer

This is the implementation of Stackelberg Decision  Transformer (STEER) for our paper accepted by ICML2024: [Sequential Asynchronous Action Coordination in Multi-Agent Systems: A Stackelberg Decision Transformer Approach](https://openreview.net/pdf?id=M3qRRkOuTN). 

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


## Citation

```
@InProceedings{pmlr-v235-zhang24au,
  title = 	 {Sequential Asynchronous Action Coordination in Multi-Agent Systems: A Stackelberg Decision Transformer Approach},
  author =       {Zhang, Bin and Mao, Hangyu and Li, Lijuan and Xu, Zhiwei and Li, Dapeng and Zhao, Rui and Fan, Guoliang},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {59559--59575},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhang24au/zhang24au.pdf},
  url = 	 {https://proceedings.mlr.press/v235/zhang24au.html},
}
```
