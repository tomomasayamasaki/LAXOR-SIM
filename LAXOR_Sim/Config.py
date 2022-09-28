import numpy as np

# Log file
LOG_FILE = 'Example.txt' # the name of a log file

# PEs
PENUMS = 256 # the number of PEs
BIT_SIZE_PE = 1024 # the number of Latch-XOR per one PE

# OR logic
ORNUMS = 256
ORBITWIDTH = 4

# BUFFER
a = 1024
BUFFERSIZE_INPUT = a #bits
BUFFERSIZE_WEIGHTS = a #bits
BUFFERSIZE_BIAS = 9 # bits

# I/O
PINS_IW = 8 # the number of pins for input or weights buffer

# BATCH NORMALIZATION
"""
    if you trained your model by theano and lasagne, please EPSILON = 0
"""
EPSILON = 0

# CIFAR10
NUM_LABELS = 10 # the number of labels

# Unit Dynamic Energy
## Computation
ENERGY_POPCOUNT = 0.00026556
ENERGY_XOR = 1.38208E-10
ENERGY_OR = 2.25811E-06
ENERGY_BNA = 0.000437255
ENERGY_COMPARISON = 5.9293E-05

## Data movement
ENERGY_DM_READ_BUFFER_IW = 2.21987E-06
ENERGY_DM_LOAD_PE = 2.20371E-08
ENERGY_DM_REAM_BUFFER_BIAS = 5.01043E-06
ENERGY_DM_LOAD_CONTROL = 5.9393E-05


# Unit Leakage Energy
# Computation
LEAK_POPCOUNT = 187.425
LEAK_XOR = 44.40087891
LEAK_OR = 24.63866016
LEAK_BNA = 791.7637969
LEAK_COMPARISON = 509.1364333

## Data movement
LEAK_DM_READ_BUFFER_IW = 23.25170117
LEAK_DM_REAM_BUFFER_BIAS = 14.56696875
LEAK_DM_LOAD_CONTROL = 31.4090625


# Clock Period
CLOCK_PERIOD_OTHERS = 0.000005 #ms
CLOCK_PERIOD_PE = (1/85/1000000)*1000 #ms
