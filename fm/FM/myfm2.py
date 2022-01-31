# author: zhangda
# date: 2021-08-21
# note: 尝试加入一阶项
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


# 二阶项
U = np.random.randn(M, K) + 0.5
V = np.random.randn(N, K) + 0.5

# 一阶项
W_user = np.random.randn(M)
W_item = np.random.randn(N)


regular = 0.001
step = 0.01


# 加载数据
def load_data(path):
	data = []
	with open(path) as input:
		for line in input:
			arr = line.strip().split("::")	
			user_id = int(arr[0])
			item_id = int(arr[1])
			rating = float(arr[2])
			data.append((user_id, item_id, rating))
	return data


# 用户对电影的评分
def evaluate(user_id, item_id):
	return sum(U[user_id] * V[item_id]) + W_user[user_id] * user_id + W_item[item_id] * item_id


# (y - y^hat) ^ 2 + reg * (u_i ^ 2 + v_j ^ 2)
def train(data):
	for (user_id, item_id, rating) in data:
		pre = evaluate(user_id, item_id)

		g_U_user = (pre - rating) * V[item_id] + regular * U[user_id]
		g_V_item = (pre - rating) * U[user_id] + regular * V[item_id]

		g_W_user = user_id
		g_W_item = item_id

		U[user_id] -= step * g_U_user
		V[item_id] -= step * g_V_item

		W_user[user_id] -= step * g_W_user;
		W_item[item_id] -= step * g_W_item;

# 计算rmse
def rmse(data):
	n = 0
	sum = 0
	for (user_id, item_id, rating) in data:
		pre = evaluate(user_id, item_id)
		sum += (pre - rating) ** 2
		n += 1
	return np.sqrt(sum/n)

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
