#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9
do
   python main.py --env-name "CartPole-v0" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 200 --num-processes 4 --lr 3e-4 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 100000 --log-dir results/cartpole_ppo_$i
   python main.py --env-name "CartPole-v0" --algo ppo_shared --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 200 --num-processes 4 --lr 3e-4 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 100000 --log-dir results/cartpole_ppo_shared_$i
done