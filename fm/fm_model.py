# author: zhangda
# date: 2021-08-22
# note: fm模型训练和评估
# 执行方式: python3 fm_model.py train_data
# TODO 如何封装为一个类？ 然后把auc的计算集成进来？

import sys
import numpy as np

# 用户编码从1到200w
M = 2000000  # user number

# item编号从1到600w
N = 6000000  # item number

# 隐向量维度
K = 10  # embedding size

regular = 0.001  # do not use regular will overfit
step = 0.01  # decide the convergence speed

U = np.random.randn(M, K)  # user embedding matrix
V = np.random.randn(N, K)  # item embedding matrix


# 加载数据
def load_data(path):
    data = []
    with open(path) as input:
        for line in input:
            arr = line.strip().split("::")
            user_id = int(arr[0])
            item_id = int(arr[1])
            rate = float(arr[2])
            data.append((user_id, item_id, rate))
    return data


# 根据用户侧和物品侧隐向量，计算预测值
def eval(user_id, item_id):
    return sum(U[user_id] * V[item_id])


# 损失函数：(y - y_hat) ^ 2 + U[k] ^ 2 + V[k] ^ 2
def train(data):
    for (user_id, item_id, rate) in data:
        # updata U[i], V[j]
        predict_value = eval(user_id, item_id)
        g_user = (predict_value - rate) * V[item_id] + regular * U[user_id]
        g_item = (predict_value - rate) * U[user_id] + regular * V[item_id]
        U[user_id] -= step * g_user
        V[item_id] -= step * g_item


# 计算预测值
def predict_list(data):
    preds = []
    for (user_id, item_id, rate) in data:
        preds.append(eval(user_id, item_id))
    return preds


# 计算均方根误差rmse
def rmse(data, pre):
    print("data.size = {}, pre.size = {}".format(len(data), len(pre)))
    sum = 0
    n = 0
    for k in range(len(data)):
        (user_id, item_id, rate) = data[k]
        predict_value = pre[k]
        sum += (predict_value - rate) ** 2
        n += 1
    return np.sqrt(sum / n)


def rmse2(data):
    sum = 0
    n = 0
    for k in range(len(data)):
        (user_id, item_id, rate) = data[k]
        predict_value = eval(user_id, item_id)
        sum += (predict_value - rate) ** 2
        n += 1
    return np.sqrt(sum / n)


def main():
    data = load_data(sys.argv[1])
    print("total data number = {}".format(len(data)))
    print(data[:10])

    # 数据集划分
    train_data = data[0:-800000]
    valid_data = data[-800000:-600000]
    test_data = data[-600000:]
    print("train_data.size = {}, valid_data.size = {}, test_data.size = {}".format(
        len(train_data), len(valid_data), len(test_data)))
    print("rmse2 = ", rmse2(data))

    # 开始训练
    for epoch in range(2):
        print("current epoch = {}, train_rmse = {}, valid_rmse = {}"
              .format(epoch, rmse2(train_data), rmse2(valid_data)))
        train(data)

    # 开始预测
    print("final test rmse = {}".format(rmse2(test_data)))
    i = 0
    for i in range(len(test_data)):
        if i > 10: break
        i += 1
        user_id, item_id, rate = test_data[i]
        predict_value = eval(user_id, item_id)
        print("label = {}, eval = {}".format(rate, predict_value))

if __name__ == '__main__':
    main()
