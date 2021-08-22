# author: zhangda
# date: 2021-08-21

import random
import sys

def load_data(path):
    lines=[]
    with open(path) as infile:
        for line in infile:
            lines.append(line)
    print("data total nums is {}".format(len(lines)))

    # shuffle
    random.shuffle(lines)

    for line in lines:
        arr = line.strip().split(',')

        if arr[3] == 'buy':
            print(arr[0] + "::" + arr[1] + "::" + '1')

        if arr[3] == 'pv':
            print(arr[0] + "::" + arr[1] + "::" + '0')

    return lines

data = load_data(sys.argv[1])

