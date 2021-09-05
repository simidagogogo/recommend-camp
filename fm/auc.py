# autho: zhangda
# date: 2021-08-21

def calAUC(prob, labels):
    f = list(zip(prob, labels))

    # 按照prob升序排列的label
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]

    # 取出所有正样本对应的索引
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]

    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1

    print("posNum = {}".format(posNum))
    print("negNum = {}".format(negNum))

    # 这个auc的计算公式怎么来的？
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc


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
print("label number = {}".format(len(y)))
print("predict number = {}".format(len(pred)))

# 计算auc
print(calAUC(pred, y))

