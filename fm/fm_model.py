# author: zhangda
# date: 2021-08-22
# note: fm模型训练和评估
# 		fm实践
# 		MovieLens 1M dataset
# 		https://grouplens.org/datasets/movielens/1m/
# 执行方式: python3 fm_model.py train_data

import sys
import numpy as np


class FmModel:
    # 用户编码从1到200w
    M = 2000000  # user number

    # item编号从1到600w
    N = 6000000  # item number

    # 隐向量维度
    K = 10  # embedding size

    U = np.random.randn(M, K)  # user embedding matrix
    V = np.random.randn(N, K)  # item embedding matrix

    def __init__(self, regular=0.001, step=0.01, epoch=1):
        self.regular = regular  # do not use regular will overfit
        self.step = step  # decide the convergence speed
        self.EPOCH = epoch

        print("M = {}, N = {}, K = {}".format(self.M, self.N, self.K))
        print("regular = {}, step = {}, epoch = {}".format(self.regular, self.step, self.EPOCH))

    # 加载数据
    def load_data(self, path):
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
    def eval(self, user_id, item_id):
        return sum(self.U[user_id] * self.V[item_id])

    # 损失函数：(y - y_hat) ^ 2 + U[k] ^ 2 + V[k] ^ 2
    def train(self, data):
        for (user_id, item_id, rate) in data:
            # updata U[i], V[j]
            predict_value = self.eval(user_id, item_id)
            g_user = (predict_value - rate) * self.V[item_id] + self.regular * self.U[user_id]
            g_item = (predict_value - rate) * self.U[user_id] + self.regular * self.V[item_id]
            self.U[user_id] -= self.step * g_user
            self.V[item_id] -= self.step * g_item

    # 计算预测值
    def predict_list(self, data):
        preds = []
        for (user_id, item_id, rate) in data:
            preds.append(self.eval(user_id, item_id))
        return preds

    # 计算均方根误差rmse
    def rmse(self, data, pre):
        print("data.size = {}, pre.size = {}".format(len(data), len(pre)))
        sum = 0
        n = 0
        for k in range(len(data)):
            (user_id, item_id, rate) = data[k]
            predict_value = pre[k]
            sum += (predict_value - rate) ** 2
            n += 1
        return np.sqrt(sum / n)

    def rmse2(self, data):
        sum = 0
        n = 0
        for k in range(len(data)):
            (user_id, item_id, rate) = data[k]
            predict_value = self.eval(user_id, item_id)
            sum += (predict_value - rate) ** 2
            n += 1
        return np.sqrt(sum / n)

    def auc(self, pre, label):
        data = list(zip(pre, label))
        label_list = [label for (pre, label) in sorted(data, key=lambda x: x[0])]

        pos_num = len([i for i in range(len(label_list)) if label_list[i] == 1])
        neg_num = len([i for i in range(len(label_list)) if label_list[i] == 0])

        pos_sum = sum([i for i in range(len(label_list)) if label_list[i] == 1])
        print("pos_num = {}, neg_num = {}, pos_sum = {}".format(pos_num, neg_num, pos_sum))
        return (pos_sum - pos_num * (pos_num - 1) / 2) / (pos_num * neg_num)

    def auc2(self, pre, label):
        data = list(zip(pre, label))
        label_list = [label for (pre, label) in sorted(data, key=lambda x: x[0])]

        pos_num = 0
        neg_num = 0
        pos_sum = 0
        for i in range(len(label_list)):
            if (label_list[i] == 1):
                pos_num += 1
                pos_sum += i  # 正样本下标从0开始
            if (label_list[i] == 0):
                neg_num += 1

        total_num = pos_num + neg_num
        ctr = pos_num / (pos_num + neg_num)
        print("total_num = {}, pos_num = {}, neg_num = {}, ctr = {}, pos_sum = {}".format(total_num, pos_num, neg_num,
                                                                                          ctr, pos_sum))
        return (pos_sum - pos_num * (pos_num - 1) / 2) / (pos_num * neg_num)

    def run(self, path):
        data = self.load_data(path)
        print("total data number = {}".format(len(data)))
        print(data[:10])

        # 数据集划分
        train_data = data[0:-800000]
        valid_data = data[-800000:-600000]
        test_data = data[-600000:]
        print("train_data.size = {}, valid_data.size = {}, test_data.size = {}".format(
            len(train_data), len(valid_data), len(test_data)))
        print("rmse2 = ", self.rmse2(data))

        train_label_list = [int(train_data[i][2]) for i in range(len(train_data))]
        valid_label_list = [int(valid_data[i][2]) for i in range(len(valid_data))]
        test_label_list = [int(test_data[i][2]) for i in range(len(test_data))]

        # 开始训练
        for epoch in range(self.EPOCH):
            train_pre_list = [self.eval(train_data[i][0], train_data[i][1]) for i in range(len(train_data))]
            train_auc = self.auc2(train_pre_list, train_label_list)
            valid_pre_list = [self.eval(valid_data[i][0], valid_data[i][1]) for i in range(len(valid_data))]
            valid_auc = self.auc2(valid_pre_list, valid_label_list)

            print("===================================")
            print("current epoch = {}, train_rmse = {}, valid_rmse = {}"
                  .format(epoch, self.rmse2(train_data), self.rmse2(valid_data)))
            print("current epoch = {}, train_auc = {}, valid_auc = {}".format(epoch, train_auc, valid_auc))
            self.train(data)

        # 开始预测
        test_pre_list = [self.eval(test_data[i][0], test_data[i][1]) for i in range(len(test_data))]
        test_auc = self.auc2(test_pre_list, test_label_list)
        print("final test rmse = {}, test_auc = {}".format(self.rmse2(test_data), test_auc))

        for i in range(len(test_data)):
            if i >= 0: break
            user_id, item_id, rate = test_data[i]
            predict_value = self.eval(user_id, item_id)
            print("label = {}, eval = {}".format(rate, predict_value))

    def test(self):
        auc = self.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
        print("auc = {}".format(auc))

if __name__ == '__main__':
    model = FmModel(epoch=10)
    # model.test()
    model.run(sys.argv[1])
    # current epoch = 9, train_rmse = 0.20807383292719187, valid_rmse = 0.20522135618991014
    # current epoch = 9, train_auc = 0.7461407058782095, valid_auc = 0.7948982665243784
    # final test rmse = 0.19101947801478666, test_auc = 0.8509381238251637
