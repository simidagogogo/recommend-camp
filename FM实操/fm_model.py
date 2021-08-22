# author: zhangda
# date: 2021-08-22

import sys
import numpy as np

# 用户编码从1到200w
M = 2000000

# item编号从1到600w
N = 6000000

# 隐向量维度
K = 10


U = np.random.randn(M, K)
V = np.random.randn(N, K)

reg = 0.001
step = 0.01


# 加载数据
def load_data(path):
	data = []
	with open(path) as input:
		for line in input:
			arr = line.strip().split("::")	
			userid = int(arr[0])
			itemid = int(arr[1])
			rating = float(arr[2])
			data.append((userid, itemid, rating))
	return data


def evaluate(userid, itemid):
	return sum(U[userid] * V[itemid])


# (y - y^hat)^2 + reg * (u_i^2 + v_j^2)
def train(data):
	for (user, item, rating) in data:
		pre = evaluate(user, item)

		g_user = (pre - rating) * V[item] + reg * U[user]
		g_item = (pre - rating) * U[user] + reg * V[item]

		U[user] -= step * g_user
		V[item] -= step * g_item


def predict(data):
    for (user, item, rating) in data:
        pre = evaluate(user, item)
        print(pre)


# 计算rmse
def rmse(data):
	num = 0
	sum = 0
	for (user, item, rating) in data:
		pre = evaluate(user, item)
		sum += (pre - rating) ** 2
		num += 1

	return np.sqrt(sum/num)		


# 开始训练
data = load_data(sys.argv[1])
print("total data number = {}".format(len(data)))
print(data[:10])


# 一共180w条样本
train_data = data[:-800000]
valid_data = data[-800000:-400000]
test_data = data[-400000:]


for i in range(10):
	print("current epoch = {}".format(i))
	print("train rmse = {}, valid rmse = {} \n".format(rmse(train_data), rmse(valid_data)))
	
	train(train_data)

print("final test rmse = {}".format(rmse(test_data)))

# 开始预测
predict(test_data)


