# author: zhangda
# date: 2021-08-21
# note:
# 		fm实践
# 		MovieLens 1M dataset
# 		https://grouplens.org/datasets/movielens/1m/

import sys
import numpy as np

# 用户编码从1到6040
M = 10000

# item编号从1到3942
N = 6000

# 隐向量维度
K = 10


U = np.random.randn(M, K) + 0.5
V = np.random.randn(N, K) + 0.5

W_user = np.random.randn(M)
W_item = np.random.randn(N)


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


# 用户对电影的评分，省略掉一次项
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

train_data = data[:-200000]
test_data = data[-200000:]

for i in range(10):
	print("current epoch = {}".format(i))
	print("train rmse = {}, test rmse = {} \n".format(rmse(train_data), rmse(test_data)))
	
	train(data)
