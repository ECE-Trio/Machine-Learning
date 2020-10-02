import os
import numpy as np

filename = 'messages.txt'

messages = np.loadtxt(filename, dtype=str, delimiter='\t')