#!/usr/bin/env python3
"""Workspace for Synthesis group "hackathon."

Implements a tiny DQN RL system which tries to generate
programs in TOYLANG (see: `env`).
"""

import os
import sys
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import INSTS, INST_LOOKUP, IO_EXAMPLE_CNT, OUTPUT_MAX, VM

# We'll just use the CPU.
device = torch.device("cpu")
# If we wanted to use a GPU when available, we could swap in:
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Constants & Hyperparameters
#
# Hyperparameters chosen via a small grid search
#
STATE_SIZE = IO_EXAMPLE_CNT * 2
ACTION_CNT = len(INSTS)
DIM_SIZE = int(os.getenv('DIM_SIZE', 32))
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = float(os.getenv('EPS_END', 0.0))
EPS_DECAY = float(os.getenv('EPS_DECAY', 10 * 1000))
TARGET_UPDATE = int(os.getenv('TARGET_UPDATE', 1000))
NUM_EPISODES = 200 * 1000


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """A collection in which to cram transitions. Training samples uniformly.

    It's important to sample "experience" relatively randomly to avoid
    overfitting to a few paths.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TinyDQN(nn.Module):

    def __init__(self):
        super(TinyDQN, self).__init__()
        self.first = nn.Linear(STATE_SIZE, DIM_SIZE)
        self.head = nn.Linear(DIM_SIZE, ACTION_CNT)
        # Could grid search over different initialization strategies; e.g:
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.uniform_(m.weight, 0.1)
        #         nn.init.constant_(m.bias, -1.0)

    def forward(self, x):
        scaled = x / OUTPUT_MAX
        return self.head(F.relu(self.first(scaled)))


policy_net = TinyDQN().to(device)
target_net = TinyDQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

eps_threshold, actions_produced = None, 0

def select_action(state):
    global eps_threshold, actions_produced
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * actions_produced / EPS_DECAY)
    actions_produced += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(ACTION_CNT)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s.reshape(1, STATE_SIZE) for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    vm = VM()
    won, tried = 0, 0
    for i_episode in range(NUM_EPISODES):
        # Initialize the environment and state
        vm.reset()
        state = vm.state.reshape(1, -1)
        for t in count():
            # Select and perform an action
            action = select_action(state)
            reward, done, next_state = vm.step(action.item())
            if done and i_episode % 1000 == 999:
                print(f"\t{reward}\t{[INST_LOOKUP[x] for x in vm.accum_prog]}")
            tried += 1
            if reward > 0.0:
                won += 1
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            assert done or next_state is not None
            if not done:
                state = next_state.reshape(1, -1)

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                break

        # Print an update
        if i_episode % 10000 == 9999:
            print(f"Episode {i_episode} of {NUM_EPISODES} - {100*i_episode/NUM_EPISODES:.1f}% complete")
            print(f"Correct: {won}/{tried}\t({100*won/(tried)}%) - (eps {eps_threshold:.2f})")
            won, tried = 0, 0

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Cooked! 🍪')

    if len(sys.argv) > 1:
        print(f"Saved policy net to {sys.argv[1]}")
        torch.save(policy_net.state_dict(), sys.argv[1])


if __name__ == '__main__':
    main()
