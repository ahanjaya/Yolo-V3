#!/usr/bin/env python3
import os

# delete rm *.txt  --> to delete all *.txt files in directory

path  = "custom_object_dataset"
files = os.listdir(path)
files = sorted(files, key = lambda x: int(x.rsplit('.', 1)[0]))

for index, file in enumerate(files):
    # print(file)
    # print(os.path.join(path, file))

    label = str(index) + '.jpg'
    os.rename(os.path.join(path, file), os.path.join(path, label))