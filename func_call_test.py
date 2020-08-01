import numpy as np
import os
import sys

func_set = set()
with open("func_call_args.txt", "r") as f:
    while (True):
        str = (f.readline())[0 : -1]

        if (str):
            func_set.add(str)

        else: break

func_list = list(func_set)
func_np = np.array(func_list).reshape(-1, 1)

with open("func_call_unique.txt", "w") as f:
    for str in func_list:
        if ("has called!!") in str:
            f.write(str + '\n')

        else:
            pass
