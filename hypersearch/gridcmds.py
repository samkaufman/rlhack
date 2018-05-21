#!/usr/bin/env python3

import random
from itertools import product

def main():
    envs = {
        'OMP_NUM_THREADS': [1],
        'EPS_END': [0.05, 0.0],
        'EPS_DECAY': [1000*1000, 100*1000, 10*1000],
        'TARGET_UPDATE': [100, 1000, 10000],
        'DIM_SIZE': [16, 32, 64],
        'NEG_REWARD': [-1.0, -0.1, -0.01]
    }
    sorted_keys = sorted(envs.keys())
    sorted_varying_keys = [k for k in sorted_keys if len(envs[k]) > 1]
    configs = list(product(*[envs[k] for k in sorted_keys]))
    random.shuffle(configs)

    for c in configs:
        label = '-'.join([str(c[sorted_keys.index(k)]) for k in sorted_varying_keys])
        assert len(c) == len(sorted_keys)
        prefix = " ".join(f"{k}={v}" for k, v in zip(sorted_keys, c))
        print(f"{prefix} python3 ./model.py model-{label}.model &> log-{label}.txt")



if __name__ == '__main__':
    main()