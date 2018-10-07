# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:26:01 2018

@author: Charles
"""

from __future__ import print_function
 
from time import time
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
 

PICTURE_PATH = "./train/"
PEOPLE = 120
FACE = 26
h = 100
w = 80

# 将摄像头收集的图像缩放切割并转换为灰度图
def pic2Grey():
	i = 1 #第i个人
	j = 1 #第j张图片
	for name in glob.glob(PICTURE_PATH+ r'*/*.jpg'):
		img = Image.open(name).convert('L')
		img = img.resize((100,100))
		#将图像缩放并切割至80*100大小
		img_resize = img.crop((10,0,90,100))
		if (i>FACE):
			i=1
			j+=1
		img_resize.save("./train/AR"+str(j).zfill(3) + "-" + str(i) +".tif")
		i+=1
        
#加载图片信息并一维化
def loadImage():
    i = 1
    while (i <= PEOPLE):
        index = 1
        while(index <= FACE):
            file_name = PICTURE_PATH + "AR" + str(i).zfill(3) + "-" + str(index) + ".tif"
            for name in glob.glob(file_name):
                img = Image.open(name)
                all_data_set.append(list(img.getdata()))#一维化图片信息
                all_data_label.append(i) #对照标签
                index += 1
        i += 1


def N_fold(clf, fold_num):
    
    print("\n-----------------------------------------\n")
    precision_average = 0.0
    kf = KFold(n_splits=fold_num, shuffle=True)
    n_components = 50
    wrong_num = 1
    test_k = []
    train_k = []
    # PCA降维
    pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(all_data_set)
    
    # PCA降维后的总数据集
    all_data_pca = pca.transform(all_data_set)
    
    # X为降维后的数据，y是对应类标签
    X = np.array(all_data_pca)
    y = np.array(all_data_label)
    
    
    
    for i in range(0, PEOPLE):
        for train, test in kf.split(np.arange(FACE)):
            test_offset = np.linspace(FACE * i, FACE * i, len(test))
            test_k.append(test + test_offset)
            train_offset = np.linspace(FACE * i, FACE * i, len(train))
            train_k.append(train + train_offset)
    
    for k in range(0, fold_num):
        test_key = np.array([])
        train_key = np.array([])
        for i in range(0, PEOPLE):
            test_key = np.append(test_key, test_k[k + fold_num * i])
            train_key = np.append(train_key, train_k[k + fold_num * i])
    
        clf = clf.fit(X[train_key.astype(np.int32)], y[train_key.astype(np.int32)])
        test_pred = clf.predict(X[test_key.astype(np.int32)])
        #print(classification_report(y[test_key.astype(np.int32)], test_pred))
        precision = 0
        for i in range(0, len(y[test_key.astype(np.int32)])):
            if (y[test_key.astype(np.int32)][i] == test_pred[i]):
                precision = precision + 1
                
            else:
                #输出错误分类信息
                print( "错误标签: ", test_pred[i] ,"\t", "错误总数 : ", wrong_num)
                wrong_num+=1
        single_precision = float(precision) / len(y[test_key.astype(np.int32)])
        precision_average = precision_average + single_precision
    precision_average = precision_average/fold_num
    return precision_average

all_data_set = []
all_data_label = []

#pic2Gray()
loadImage()
 
X = np.array(all_data_set)
y = np.array(all_data_label)
n_samples,n_features = X.shape
n_classes = len(np.unique(y))
target_names = []

for i in range(1,121):
    names = "person" + str(i)
    target_names.append(names)
 
print("\n--------------数据集信息:-----------------\n")
print("总样本: %d" % n_samples)
print("总特征数: %d" % n_features)
print("总人数: %d" % n_classes)
 
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
 
n_components = 50

print("n_components: %d\n" % n_components)

# -------------PCA降维----------------------
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
all_data_pca = pca.transform(all_data_set)

 
## 输出特征脸-------------------------------------------------
#eigenfaces = pca.components_.reshape((n_components, 100, 80))
#plt.figure("Eigenfaces")
#for i in range(1, 51):
#    plt.subplot(5, 10, i).imshow(eigenfaces[i-1], cmap="gray")
#    plt.xticks(())
#    plt.yticks(())
#
#plt.show()
#
##----------------------------------------------------------------

#--------------网格搜索穷举最优  C , gamma  ---------------
print("---------网格搜索C,gamma最优解-----------\n")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#kernel采用高斯核函数
#class_weight='balanced'表示调整各类别权重，权重与该类中样本数成反比，
#防止模型过于拟合某个样本数量过大的类
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
#----------------output------------------
#SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
#-------------------------------------------

#将输出的 C与gamma直接带入，节约时间

param_grid = {'C': [1e3],
              'gamma': [0.001], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)

print("Best estimator:",clf.best_estimator_)
print("done in %0.3fs" % (time() - t0))

#---------k折对识别率的影响---------------
#t0=time()
#plt.figure(1)
#x_label = []
#y_label = []
#for m in range(2,11):
#    acc = 0
#    x_label.append(m)
#    #for n in range(0, 10):
#    acc = acc + N_fold(clf, m)*100
#    y_label.append(acc)
#    plt.plot(x_label, y_label)
#
#print("done in %0.3fs" % (time() - t0))
#plt.xlabel("fold_num")
#plt.ylabel("Precision")
#plt.title('Different fold_num')
#plt.legend()
#plt.show()
#------------------------------------------------





fold_num = 5
print("\n", fold_num, "折重交叉平均准确率为%.2f"%(N_fold(clf, fold_num)*100),"%\n")



# -----------------------检验-----------------------------

# print("-------------------预测结果-------------------\n")
# #t0 = time()
# y_pred = clf.predict(X_test_pca)
 
# print(classification_report(y_test, y_pred, target_names=target_names))
# #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
 
# #print("done in %0.3fs" % (time() - t0))