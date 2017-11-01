
"""
数据集是使用的周克华教授编写的机器学习实战中的西瓜例子，在编写代码之前对数据集属性进行标注：
色泽：0代表青绿，1代表乌黑，2代表浅白
根蒂：0代表蜷缩，1代表稍蜷，2代表硬挺
敲声：0代表浊响，1代表沉闷，2代表清脆
纹理：0代表清晰，1代表稍糊，2代表模糊
脐部：0代表凹陷，1代表稍凹，2代表平坦
触感：0代表赢滑，1代表软粘
类别：是否为好瓜，‘yes’代表好瓜，‘no’代表不是好瓜
Author:
    Winter Fu
Modify:
    2017-06-06
"""
# -*- coding: UTF-8 -*-
# 构建数据集
def createDataSet():
    dataSet = [ [0, 0, 0, 0, 0, 0, 'yes'],
                [1, 0, 1, 0, 0, 0, 'yes'],
                [1, 0, 0, 0, 0, 0, 'yes'],
                [0, 0, 1, 0, 0, 0, 'yes'],
                [2, 0, 0, 0, 0, 0, 'yes'],
                [0, 1, 0, 0, 1, 1, 'yes'],
                [1, 1, 0, 1, 1, 1, 'yes'],
                [1, 1, 0, 0, 1, 0, 'yes'],
                [1, 1, 1, 1, 1, 0, 'no'],
                [0, 2, 2, 0, 2, 1, 'no'],
                [2, 2, 2, 2, 2, 0, 'no'],
                [2, 0, 0, 2, 2, 1, 'no'],
                [0, 1, 0, 1, 0, 0, 'no'],
                [2, 1, 1, 1, 0, 0, 'no'],
                [1, 1, 0, 0, 1, 1, 'no'],
                [2, 0, 0, 2, 2, 0, 'no'],
                [0, 0, 1, 1, 1, 0, 'no']
              ]
    labels = ['色泽', '根蒂','敲声','纹理','脐部','触感']
    return dataSet, labels

# 计算经验熵
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #计算实例个数
    labelCounts = { }          #建立计数字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= prob*log(prob, 2)
    return shannoEnt
"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
"""
def splitDataSet(dataSet, axis, value):       
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  
import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

"""
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
    
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                    #特征数量
    baseEntropy = calcShannonEnt(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                         #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                         #创建set集合{},元素不可重复
        newEntropy = 0.0                                  #经验条件熵
        for value in uniqueVals:                         #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)         #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))           #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                     #信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))            #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                             #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的特征的索引值
    return bestFeature                                          #返回信息增益最大的特征的索引值
# 构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]          #创建包含数据集所有类标签的列表变量
    if classList.count(classList[0]) == len(classList):       #所有的类标签完全相同时直接返回该类标签
        return classList[0]          
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}                               #用字典类型来存储树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

if __name__ == '__main__':
    dataSet, features = createDataSet()
    myTree = createTree(dataSet, features)
    print(myTree)