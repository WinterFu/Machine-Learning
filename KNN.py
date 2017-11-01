# -*- coding: utf-8 -*-
"""
数据集是使用的sklern库中的Iris数据集
特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
类别：Setosa, Versicolour, Virginica

使用KNN算法来对Iris数据集进行分类
Author:
    Winter Fu
Modify:
    2017-11-01
"""

import numpy as np 
import random
import heapq
from sklearn import datasets

#对数据进行预处理，分成训练集和测试集
def Processing():
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	#加载Iris数据集
	iris = datasets.load_iris()
	data = iris.data[:, 0:4]          #获取素有数据，不含标签
	label = iris.target               #获取所有标签

	#获取0-149中任意不重复的10个数
	test_list = np.sort(random.sample(range(150), 10))

	#对数据集进行划分
	for i in range(150):
		if i in test_list:
			test_data.append(data[i])
			test_labels.append(label[i])
		else:
			train_data.append(data[i])
			train_labels.append(label[i])

	train_data = np.mat(train_data)   
	train_labels = np.mat(train_labels).transpose()  #进行矩阵的转换
	test_data = np.mat(test_data)   #转换为矩阵类型
	test_labels = np.mat(test_labels).transpose()

	return train_data, train_labels, test_data, test_labels

def Predict(train_data, train_labels, test_data, n):
	k = n                      #设置初始的K值
	prediction = []
	
	#遍历整个测试集
	for test_vec in test_data:
		dist_list = []             #用于存放测试数据与训练数据的距离
		knn_lsit = []              #用来存放前k个最近邻点

		#遍历整个训练集中的样本特征向量和标签
		for i in range(len(train_labels)):
			label_vec = train_labels[i]
			train_vec = train_data[i]

			dist = np.linalg.norm(train_vec - test_vec)       #计算测试样本和训练样本之间的欧氏距离
			dist_list.append((dist, label_vec))

		knn_lsit = heapq.nsmallest(k, dist_list, key = lambda x: x[0])    #使用Python的对结构取k个最近邻

		#统计选票
		class_total = 3
		class_count = [0 for i in range(class_total)]
		for dist, label in knn_lsit:
			label = label.tolist()        #将矩阵转换成列表
			class_count[label[0][0]] += 1
	
		#找出最大选票
		vote_max = np.argmax(class_count)
		prediction.append(vote_max)

	return prediction

if __name__ == "__main__":
	k = 8
	train_data, train_labels, test_data, test_labels = Processing()
	predict = Predict(train_data, train_labels, test_data, k)
	test_labels = test_labels.tolist()

	error_count = 0
	for i in range(10):
		print("test_data: ", test_labels[i][0], "prediction: ", predict[i] )
		if test_labels[i][0] != predict[i]:
			error_count += 1

	print("error_rate: ", error_count/10)
