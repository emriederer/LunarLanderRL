# Lunar Lander Reinforcement Learning

<p align="center">
  <img src="/assets/result.gif" alt="Lunar Lander Neuroevolution in action" width="300"/>
</p>

This project is dedicated to the development and comparison of different AI models trained to play the Lunar Lander game. The goal is to successfully land a lunar module on the surface of the moon. The repository includes three distinct approaches: neuroevolution, DQN (Deep Q-Network), and Double DQN.

## Models

### Neuroevolution
Neuroevolution is a form of artificial intelligence that uses evolutionary algorithms to generate neural networks, mimicking biological evolution. This model evolves through generations to optimize its landing strategy.

### DQN (Deep Q-Network)
DQN is a reinforcement learning algorithm that combines Q-Learning with deep neural networks. This model learns through trial and error, using a reward system to make better landing decisions over time.

### Double DQN
Double DQN is an improvement over the standard DQN algorithm that reduces the overestimation of action values. This model aims to achieve more stable and reliable learning outcomes.

## Results and Comparison

The training progress and results can be monitored through loss graphs which are saved in the `assets` directory. The images below are limited to 500px in width for consistency in presentation.

### Neuroevolution Loss
<img src="/assets/neuroevolution.png" alt="Neuroevolution Loss" width="500"/>

### DQN Loss
<img src="/assets/dqn.png" alt="DQN Loss" width="500"/>

### Double DQN Loss
<img src="/assets/ddqn.png" alt="Double DQN Loss" width="500"/>
