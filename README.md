# KISSDE

This is keep-it-simple-and-stupid realization of
[Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde_pytorch).
The whole model is written in pure PyTorch and made as self-explanatory as possible.

## Model parts
This realization contains a basic convolutional U-Net-like score approximation model
and predictor-corrector as a sampler. The whole code is based on a different parts of
the mentioned repo. Some major improvements like EMA of weights are implemented, leading
to reproducing nearly SotA results on CIFAR10, while using tutorial-like architecture.

## How to train
Firstly, build docker file with
```
docker build -t score_sde .
```
Then you can specify your wandb key in run.yaml and run training with [crafting](https://pypi.org/project/crafting/0.1/)
```
crafting run.yaml
```
