import numpy as np 

def Percep(X_train, Y_train):
	#获取数据维度
	m, n = np.shape(X_train)

	#初始化参数
	w = np.zeros((n, 1))
	b = 0
	lr = 1
	while True:
		counter = m   #记录被修改的次数，当不再发生修改即没有被误分点时跳出while循环

		for i in range(m):
			result = Y_train[i]*(np.dot(X_train[i], w) + b)

			if result <= 0:
				counter -= 1
				#更新参数w, b
				for j in range(n):
					w[j] = w[j] + lr*Y_train[i]*X_train[i][j]
				b = b + lr*Y_train[i]

			print("w = ", w, "b = ", b)

		if counter == m:
			break

	return w, b

def predict(x_test, w_opt, b_opt):
	m, n = np.shape(x_test)

	prediction = []

	for k in range(m):
		y_pred = np.sign(np.dot(x_test[k], w_opt) + b_opt)
		prediction.append(y_pred)
	return prediction

if __name__ == "__main__":
	train_data = np.array([[3, 3], [4, 3], [1, 1]])
	train_labels = np.array([1, 1, -1])
	test_data = np.array([[1, 0]])

	w_opt, b_opt = Percep(train_data, train_labels)
	result = predict(test_data, w_opt, b_opt)
	print(result)