#!/usr/bin/env python3

from glob import glob
import os
import sys
import random

if len(sys.argv) != 2:
    print("Usage: {} <folder>".format(sys.argv[0]))
    sys.exit(1)

folder        = sys.argv[1]
files         = glob(folder + "/*.jpg") # list all files with *.jpg extension in directory
total_size    = len(files)
indices       = list(range(total_size))

# splitting data
split_percent = 0.05
valid_n       = int(total_size * split_percent)
train_n       = total_size - valid_n
print("Total of: {} images, will use {} for training and {} for validation.".format(total_size, train_n, valid_n))

# random distribute
random.shuffle(indices)

# custom_train.txt
with open("custom_train.txt", "w") as f:
    for i in indices[:train_n]:
        f.write( os.path.abspath(files[i]) + '\n' )

# custom_valid.txt
with open("custom_valid.txt", "w") as f:
    for i in indices[train_n:]:
        f.write( os.path.abspath(files[i]) + '\n' )

print("Finish write files into {} and {}".format('custom_train.txt', 'custom_valid.txt'))