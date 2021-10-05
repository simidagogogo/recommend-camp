# author: zhangda
# date: 2021-08-21
# note: 将原始的日志数据处理为libfm的输入特征格式
#       可能需要对用户和item进行重新编码

# 1 12345:1 23456:1

import random
import sys

def load_data(path):
    lines=[]
    with open(path) as infile:
        for line in infile:
            lines.append(line)

    # shuffle
    random.shuffle(lines)

    for line in lines:
        arr = line.strip().split(',')

        if arr[3] == 'buy':
            print('1 ' + arr[0] + ':1 ' + arr[1] + ':1 ' + arr[2] + ':1')

        if arr[3] == 'pv':
            print('0 ' + arr[0] + ':1 ' + arr[1] + ':1 ' + arr[2] + ':1')

    return lines

data = load_data(sys.argv[1])
print("data total nums is {}".format(len(data)))
