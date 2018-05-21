#!/usr/bin/env python3
"""Programs in TOYLANG always have one integer
input and one integer output. Instructions are:
DOUBLE, SQUARE, ADD1, SUB1, MAX0, RETURN. Programs
are 7 instructions or shorter. Inputs are all in [-8, 8]
and target outputs are in [-128, 128]. Registers
are always in the range [-16384, 16384]. 
"""

import os
import random
import torch
from copy import deepcopy

IO_EXAMPLE_CNT = 3
INPUT_MIN, INPUT_MAX = -8, 8
OUTPUT_MIN, OUTPUT_MAX = -128, 128
REGISTER_MAX = 16384
MAX_PROG_LEN = 4
NEG_REWARD = float(os.getenv("NEG_REWARD", -1.0))

INSTS = {
    'DOUBLE': lambda x: 2*x,
    'SQUARE': lambda x: x**2,
    'ADD1': lambda x: x + 1,
    'SUB1': lambda x: x - 1,
    'MAX0': lambda x: max(0, x),
    'RETURN': lambda x: x,
}
INST_LOOKUP = ['ADD1', 'SUB1', 'MAX0', 'DOUBLE', 'SQUARE', 'RETURN']
assert set(INSTS.keys()) == set(INST_LOOKUP)


def evaluate(inp, inst):
    """Given an input integer and an instruction, return the result.

    Also accepts iterables of instructions.

    >>> evaluate(3, 'SQUARE')
    9
    >>> evaluate(0, 'SUB1')
    -1
    >>> evaluate(-1, 'MAX0')
    0
    >>> evaluate(1, ['ADD1', 'ADD1'])
    3
    """
    if isinstance(inst, int):
        return evaluate(inp, INST_LOOKUP[inst])
    elif isinstance(inst, str):
        return INSTS[inst](inp)
    else:
        inst = iter(inst)
        try:
            head_op = next(inst)
        except StopIteration:
            return inp
        return evaluate(INSTS[head_op](inp), inst)


def sample_toy(maxlen=MAX_PROG_LEN):
    for _ in range(random.randint(1, maxlen - 1)):
        yield random.choice([k for k in INSTS.keys() if k != 'RETURN'])
    yield 'RETURN'


class VM:

    def __init__(self, skip_reset=False):
        super().__init__()
        if not skip_reset:
            self.reset()

    def reset(self):
        self.returned = False
        self.accum_prog = []
        self.target_prog = list(sample_toy())
        inp_rng = [x + 1 if x >= 0 else x for x in range(INPUT_MIN, INPUT_MAX)]
        while 1:
            self.registers = [0] + sorted(random.sample(inp_rng, IO_EXAMPLE_CNT - 1))
            self.output_targets = [evaluate(i, self.target_prog) for i in self.registers]
            if all(t >= OUTPUT_MIN and t <= OUTPUT_MAX for t in self.output_targets):
                break

    def step(self, op):
        if self.returned:
            raise Exception("Already returned")
        if isinstance(op, str):
            op = INST_LOOKUP.index(op)
        self.accum_prog.append(op)
        self.registers = [evaluate(i, op) for i in self.registers]

        reward, done = 0.0, False
        if any(abs(r) > REGISTER_MAX for r in self.registers):
            # An "overflow"
            reward, done = NEG_REWARD, True
            self.returned = True
        elif INST_LOOKUP[op] == 'RETURN':
            reward, done = 1.0, True
            for r, t in zip(self.registers, self.output_targets):
                if r != t:
                    reward = NEG_REWARD
            self.returned = True
        elif len(self.accum_prog) == MAX_PROG_LEN:
            reward, done = NEG_REWARD, True
            self.returned = True
        return reward, done, self.state

    @property
    def state(self):
        if self.returned:
            return None
        try:
            return torch.tensor(self.registers + self.output_targets, dtype=torch.float32)
        except:
            print(self.registers)
            print(self.output_targets)
            raise

    def __copy__(self):
        n = VM(skip_reset=True)
        n.returned = self.returned
        n.accum_prog = self.accum_prog
        n.target_prog = self.target_prog
        n.registers = self.registers
        n.output_targets = self.output_targets
        return n

    def __deepcopy__(self, memo):
        n = VM(skip_reset=True)
        n.returned = self.returned
        n.accum_prog = deepcopy(self.accum_prog, memo)
        n.target_prog = deepcopy(self.target_prog, memo)
        n.registers = deepcopy(self.registers, memo)
        n.output_targets = deepcopy(self.output_targets, memo)
        return n


def _solve_via_search(vm):
    from itertools import product
    non_ret_op_idxs = [x for x in range(len(INST_LOOKUP))
                       if INST_LOOKUP[x] != 'RETURN']
    orig_vm = vm
    for l in range(1, MAX_PROG_LEN):
        for prog in product(non_ret_op_idxs, repeat=l):
            prog = prog + (INST_LOOKUP.index('RETURN'),)
            vm = deepcopy(orig_vm)
            for i in prog:
                reward, done, _ = vm.step(i)
                if done and reward == 1.0:
                    print(f"Solution: {[INST_LOOKUP[i] for i in prog]}")
                if done:
                    break


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    print("Kicking the \"virtual machine\":")
    vm = VM()
    print(vm.target_prog)
    print(vm.registers)
    print(vm.output_targets)
    print(vm.state)

    print("")
    print("Solving via search:")
    _solve_via_search(vm)
