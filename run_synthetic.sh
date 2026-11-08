#!/bin/bash
set -e
for env in GoalPuddle-5x5 TrapGrid-9x9 XYTradeoff-9x9; do
    for seed in 10 20 30 40 50 60 70 80 90 100; do
        python run_synthetic.py --env "$env" --seed "$seed" --out results/synthetic.csv
    done
done
