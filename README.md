# DDPG Mountain Car Continuous

### Wang Huisheng, 2022.11.5

---

## Requirements

- Numpy
- Pickle
- Tensorflow
- Open AI Gym


## How to run

There is a Constant DEVICE = '/cpu:0', you if you have a gpu you can set it to DEVICE = '/gpu:0' and it will use tensorflow for training. 
To run the algorithm you can do: 

```
python main.py
```

## Sources

- [DDPG_MountainCar](https://github.com/IgnacioCarlucho/DDPG_MountainCar/blob/master/mountain.py)
- [Gym Mountain Car Continuous](https://github.com/openai/gym/wiki/MountainCarContinuous-v0)
- [DDPG Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Implementation of the Ornstein-Uhlenbeck Noise](https://github.com/openai/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py)
