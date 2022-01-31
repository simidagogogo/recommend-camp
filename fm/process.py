# author: zhangda
# date: 2021-08-21
# note: 对原始数据文件data_1.csv进行处理，转换为user_id::item_id::label数据格式
# 执行方式: python3 process.py > train_data

import random
import sys


def load_data(path):
    lines = []
    with open(path) as infile:
        for line in infile:
            lines.append(line)
    print("data total nums is {}".format(len(lines)))

    # shuffle
    random.shuffle(lines)

    for line in lines:
        arr = line.strip().split(',')

        if arr[3] == 'pv':
            print(arr[0] + "::" + arr[1] + "::" + '0')

        if arr[3] == 'buy':
            print(arr[0] + "::" + arr[1] + "::" + '1')

    return lines

if __name__ == '__main__':
    data = load_data(sys.argv[1])