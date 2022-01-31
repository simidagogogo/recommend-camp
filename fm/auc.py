# autho: zhangda
# date: 2021-08-21
# note: 计算auc

import sys


def calAUC(prob, labels):
    # 将预测值和label拼在一起，形成二元组
    f = list(zip(prob, labels))

    # 按照prob升序排列，获取label序列
    rank = [label for pre, label in sorted(f, key=lambda x: x[0])]

    # 取出所有正样本对应的索引
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]

    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    print("total = {}, posNum = {}, negNum = {}".format(posNum + negNum, posNum, negNum))

    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc


if __name__ == '__main__':
    # 预测值
    pred = []
    with open("score", 'r') as infile:
        for line in infile:
            pred.append(float(line.strip()))

    # 真实值
    y = []
    with open("label", 'r') as infile:
        for line in infile:
            y.append(int(line.strip().split('::')[2]))

    # 真实样本个数，必须和预测样本个数一致
    print("label number = {}, predict number = {}".format(len(y), len(pred)))
    assert len(y) == len(pred)

    # 计算auc
    auc = calAUC(pred, y)
    print(auc)
