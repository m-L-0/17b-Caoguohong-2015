# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2017/11/14 下午3:13
# # @Author  : tudoudou
# # @File    : test.py
# # @Software: PyCharm
#
# import os
# import gzip
# import numpy as np
# import tensorflow as tf
# from PIL import Image
#
#
# def load_mnist(path='./data/', kind='train'):
#     """
#     加載mnist數據集
#     :param path: 傳入路徑
#     :param kind: 類別
#     :return:
#     """
#     labels_path = os.path.join(path, '{0}-labels-idx1-ubyte.gz'.format(kind))
#     images_path = os.path.join(path, '{0}-images-idx3-ubyte.gz'.format(kind))
#
#     with gzip.open(labels_path, 'rb') as lbpath:
#         labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
#
#     with gzip.open(images_path, 'rb') as imgpath:
#         images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
#
#     return images, labels
#
#
# imgs, labs = load_mnist()
#
# writer = tf.python_io.TFRecordWriter("train.tfrecords")
# i = 0
# for img in imgs:
#     i += 1
#     if i % 1000==0:
#         print("已处理{0}张".format(i))
#     img = Image.fromarray(img.reshape(28, 28)).tobytes()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labs[i]])),
#         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
#     }))  # example对象对label和image数据进行封装
#     writer.write(example.SerializeToString())  # 序列化为字符串
# writer.close()


# import numpy as np
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets('./data/',validation_size=50)

# print(data.validation.images)

# print(data)
# print(len(data.train.next_batch(60000)[0]))
# print(len(data.validation.next_batch(10000)[0]))
# print(len(data.test.next_batch(10000)[0]))

# from PIL import Image


# imgs, labs = load_mnist()

# def write_tfrecord(lable, data):
#     writer = tf.python_io.TFRecordWriter(lable + ".tfrecords")
#     num = len(data[0])
#     for i in range(num):
#         if (i + 1) % 10000 == 0:
#             print("以處理{0}數據集{1}張".format(lable, i + 1))
#         img = data[0][i]
#
#         # print(img)
#         # print("++++++++++++++++++++++++++++++++++++++++")
#         # print(img.reshape(28, 28))
#         # print("========================================")
#         # img = img.reshape(28, 28)
#         print([img][0])
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[data[1][i]])),
#             'img_val': tf.train.Feature(float_list=tf.train.FloatList(value=[img][0]))
#         }))  # example對象 對label及img_val 進行封裝
#         writer.write(example.SerializeToString())  # 序列化字符串?
#         i += 1
#     print("{0}數據集處理完成".format(lable))
#     writer.close()


# write_tfrecord("train",data.train.next_batch(60000))
# write_tfrecord("validation",data.validation.next_batch(10))
# write_tfrecord("test",data.test.next_batch(10000))
# write_tfrecord("demo",data.validation.next_batch(10))


# import tensorflow as tf
# from PIL import Image



# filename_queue = tf.train.string_input_producer(["demo.tfrecords"]) #读入流中
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
# features = tf.parse_single_example(serialized_example,
#                                    features={
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'img_val' : tf.FixedLenFeature([28,28], tf.float32),
#                                    })  #取出包含image和label的feature对象
# image = tf.cast(features['img_val'], tf.float64)
# # image = tf.reshape(image, [28, 28])
# label = tf.cast(features['label'], tf.int32)
#
# print(image)
# print(label)
#
#
# with tf.Session() as sess: #开始一个会话
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     coord=tf.train.Coordinator()
#     threads= tf.train.start_queue_runners(coord=coord)
#     for i in range(20):
#         example, l = sess.run([image,label])#在会话中取出image和label
#         example*=255
#         # print(example)
#         example = np.array(example, dtype='uint8')
#
#         # example=tf.reshape(example,[28,28])
#         # example=example.reshape(28,28)
#         img=Image.fromarray(example)#这里Image是之前提到的
#
#         img.save(str(i)+'_''Label_'+str(l)+'.bmp','bmp')#存下图片
#         # print(example, l)
#     coord.request_stop()
#     coord.join(threads)
#
#
#



#
#

# import numpy as np
# import tensorflow as tf
#
# def read_tfrecord(filename,tensor=[1,784],num=5000):
#     filename_queue = tf.train.string_input_producer([filename]) #读入流中
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_val' : tf.FixedLenFeature(tensor, tf.float32),
#                                        })  #取出包含image和label的feature对象
#     image = tf.cast(features['img_val'], tf.float64)
#     label = tf.cast(features['label'], tf.int32)
#     images=[];labels=[]
#     with tf.Session() as sess:
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
#         coord=tf.train.Coordinator()
#         threads= tf.train.start_queue_runners(coord=coord)
#         for i in range(num):
#             example, l = sess.run([image, label])
#             images.append(np.array(example[0]))
#             tem=np.zeros((1,10))
#             tem[0][l]=1.0
#             labels.append(tem[0])
#             del tem
#         coord.request_stop()
#         coord.join(threads=threads)
#     images=np.array(images)
#     labels=np.array(labels)
#     return images,labels
#
#
# # img,lab=read_tfrecord(filename='test.tfrecords',tensor=[1,784],num=1)
# # print(img[0])
# # print(lab)
# #
# # from tensorflow.examples.tutorials.mnist import input_data
# # mnist = input_data.read_data_sets('./data/', one_hot=True)
# # train_x, train_y = mnist.train.next_batch(1)
# #
# # print(train_x)
# # print(train_y)
#
#
# def loadMNIST():
#     from tensorflow.examples.tutorials.mnist import input_data
#     mnist = input_data.read_data_sets('./data/', one_hot=True)
#
#
#     return mnist
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./data/', one_hot=True)
# train_x, train_y = mnist.train.next_batch(6)
# print(train_x[0])
# print("-------------------------------------------")
#
# def KNN():
#     # train_x, train_y = mnist.train.next_batch(60000)
#     # test_x, test_y = mnist.test.next_batch(5000)
#     train_x,train_y=read_tfrecord(filename='train.tfrecords',tensor=[1,784],num=600)
#     test_x, test_y =read_tfrecord(filename='test.tfrecords',tensor=[1,784],num=500)
#     # 數據二值化處理
#     for i in range(len(train_x)):
#         for j in range(len(train_x[0])):
#             if train_x[i][j]>0:
#                 train_x[i][j]=1.0
#     # print(train_x[0])
#     for i in range(len(test_x)):
#         for j in range(len(test_x[0])):
#             if test_x[i][j] > 0:
#                 test_x[i][j] = 1.0
#     xtr = tf.placeholder(tf.float32, [None, 784])
#     xte = tf.placeholder(tf.float32, [784])
#     distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)), 2), reduction_indices=1))
#
#     pred = tf.argmin(distance, 0)
#
#     init = tf.global_variables_initializer()
#
#     sess = tf.Session()
#     sess.run(init)
#
#     right = 0
#     for i in range(500):
#         if i % 250==0 and i !=0:
#             print("已处理 {0}，正确率为{1}".format(i,right/i))
#         ansIndex = sess.run(pred, {xtr: train_x, xte: test_x[i, :]})
#         # print('prediction is ', str(np.where(train_y[ansIndex]==np.max(train_y[ansIndex]))))
#         # print('true value is ', str(np.where(test_y[i]==np.max(test_y[i]))))
#         if np.argmax(test_y[i]) == np.argmax(train_y[ansIndex]):
#             right += 1.0
#     accracy = right / 500.0
#     print(accracy)
#
#
# if __name__ == "__main__":
#     # mnist = loadMNIST()
#     KNN()
#








# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.datasets.samples_generator import make_circles
#
# K = 4 # 类别数目
# MAX_ITERS = 1000 # 最大迭代次数
# N = 200 # 样本点数目
#
# centers = [[-2, -2], [-2, 1.5], [1.5, -2], [2, 1.5]] # 簇中心
#
# # 生成人工数据集
# #data, features = make_circles(n_samples=200, shuffle=True, noise=0.1, factor=0.4)
# data, features = make_blobs(n_samples=N, centers=centers, n_features = 2, cluster_std=0.8, shuffle=False, random_state=42)
# print(data)
# print(features)
#
# def clusterMean(data, id, num):
#     total = tf.unsorted_segment_sum(data, id, num) # 第一个参数是tensor，第二个参数是簇标签，第三个是簇数目
#     count = tf.unsorted_segment_sum(tf.ones_like(data), id, num)
#     return total/count
#
# # 构建graph
# points = tf.Variable(data)
# cluster = tf.Variable(tf.zeros([N], dtype=tf.int64))
# centers = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))# 将原始数据前k个点当做初始中心
# repCenters = tf.reshape(tf.tile(centers, [N, 1]), [N, K, 2]) # 复制操作，便于矩阵批量计算距离
# repPoints = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
# sumSqure = tf.reduce_sum(tf.square(repCenters-repPoints), reduction_indices=2) # 计算距离
# bestCenter = tf.argmin(sumSqure, axis=1)  # 寻找最近的簇中心
# change = tf.reduce_any(tf.not_equal(bestCenter, cluster)) # 检测簇中心是否还在变化
# means = clusterMean(points, bestCenter, K)  # 计算簇内均值
# # 将粗内均值变成新的簇中心，同时分类结果也要更新
# with tf.control_dependencies([change]):
#     update = tf.group(centers.assign(means),cluster.assign(bestCenter)) # 复制函数
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     changed = True
#     iterNum = 0
#     while changed and iterNum < MAX_ITERS:
#         iterNum += 1
#         # 运行graph
#         [changed, _] = sess.run([change, update])
#         [centersArr, clusterArr] = sess.run([centers, cluster])
#         print(clusterArr)
#         print(centersArr)
#
#         # 显示图像
#         fig, ax = plt.subplots()
#         ax.scatter(data.transpose()[0], data.transpose()[1], marker='o', s=100, c=clusterArr)
#         plt.plot()
#         plt.show()













# # % matplotlib inline
# import tensorflow as tf
# import os
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects
# import matplotlib
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("./data/", one_hot=True)
# current_dir = os.getcwd()
# sess = tf.InteractiveSession()
#
#
# def weight_variable(shape, name):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial, name)
#
#
# def bias_variable(shape, name):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial, name)
#
#
# def conv2d(X, W):
#     return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(X):
#     return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
# def add_layer(X, W, B):
#     h_conv = tf.nn.relu(conv2d(X, W) + B)
#     return max_pool_2x2(h_conv)
#
#
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# layer1 = add_layer(x_image, weight_variable([5, 5, 1, 32], "w_conv1"), bias_variable([32], "b_conv1"))
#
# layer2 = tf.nn.relu(conv2d(layer1, weight_variable([5, 5, 32, 48], "w_conv2")) + bias_variable([48], "b_conv2"))
#
# layer3 = add_layer(layer2, weight_variable([5, 5, 48, 64], "w_conv3"), bias_variable([64], "b_conv3"))
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024], "w_fc1")
# b_fc1 = bias_variable([1024], "b_fc1")
# h_pool2_flat = tf.reshape(layer3, [-1, 7 * 7 * 64])
#
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# W_fc2 = weight_variable([1024, 10], "w_fc2")
# b_fc2 = bias_variable([10], "b_fc2")
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# saver = tf.train.Saver()
#
# saver.restore(sess, os.path.join(current_dir, "model/mnist_cnn_3_layer/model.ckpt"))
#
# test_size = 5000
# test_data = mnist.test.images[0:test_size, :]
# test_label = mnist.test.labels[0:test_size, :]
# test_label_index = np.argmax(test_label, axis = 1)
#
#
# def tsne(X, n_components):
#     model = TSNE(n_components=2, perplexity=40)
#     return model.fit_transform(X)
#
#
# def plot_scatter(x, labels, title, txt=False):
#     plt.title(title)
#     ax = plt.subplot()
#     ax.scatter(x[:, 0], x[:, 1], c=labels)
#     txts = []
#     if txt:
#         for i in range(10):
#             xtext, ytext = np.median(x[labels == i, :], axis=0)
#             txt = ax.text(xtext, ytext, str(i), fontsize=24)
#             txt.set_path_effects([
#                 PathEffects.Stroke(linewidth=5, foreground="w"),
#                 PathEffects.Normal()])
#             txts.append(txt)
#     plt.show()
#
#
# fc2_tsne = tsne(y_conv.eval(feed_dict={x: test_data, keep_prob: 1.0}), 2)
# plot_scatter(fc2_tsne, test_label_index, "fc layer2 with tsne", txt=True)




# import numpy as np
# from sklearn.manifold import TSNE
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('./data/', one_hot=True)
# X_embedded = TSNE(n_components=2).fit_transform(mnist.train.next_batch(1000)[0])
# a=[]
# b=[]
# for i,j in X_embedded:
#     a.append(i)
#     b.append(j)
# print(X_embedded)


# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.manifold import TSNE
#
#
# def loadmnist():
#     from tensorflow.examples.tutorials.mnist import input_data
#     return input_data.read_data_sets('./data/')
# data = loadmnist()
#
# mnist = data.train.next_batch(10)
# X_embedded = TSNE(n_components=2, learning_rate=800.0, early_exaggeration=50.0, n_iter=5000).fit_transform(mnist[0])
#
#
# print(X_embedded)
# li=[];x=[];y=[];z=[]
# for i in X_embedded:
#     x,y,z=i
#     li.append([x,y,z])
#
# print(li)



# import numpy as np
#
#
# class KMeans(object):
#     """
#     - 参数
#         n_clusters:
#             聚类个数，即k
#         initCent:
#             质心初始化方式，可选"random"或指定一个具体的array,默认random，即随机初始化
#         max_iter:
#             最大迭代次数
#     """
#     def __init__(self, n_clusters=5, initCent='random', max_iter=300):
#         if hasattr(initCent, '__array__'):
#             n_clusters = initCent.shape[0]
#             self.centroids = np.asarray(initCent, dtype=np.float)
#         else:
#             self.centroids = None
#
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.initCent = initCent
#         self.clusterAssment = None
#         self.labels = None
#         self.sse = None
#
#         # 计算两点的欧式距离
#
#     def _distEclud(self, vecA, vecB):
#         return np.linalg.norm(vecA - vecB)
#
#     # 随机选取k个质心,必须在数据集的边界内
#     def _randCent(self, X, k):
#         n = X.shape[1]  # 特征维数
#         centroids = np.empty((k, n))  # k*n的矩阵，用于存储质心
#         for j in range(n):  # 产生k个质心，一维一维地随机初始化
#             minJ = min(X[:, j])
#             rangeJ = float(max(X[:, j]) - minJ)
#             centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
#         return centroids
#
#     def fit(self, X):
#         # 类型检查
#         if not isinstance(X, np.ndarray):
#             try:
#                 X = np.asarray(X)
#             except:
#                 raise TypeError("numpy.ndarray required for X")
#
#         m = X.shape[0]  # m代表样本数量
#         self.clusterAssment = np.empty((m, 2))  # m*2的矩阵，第一列存储样本点所属的族的索引值，
#         # 第二列存储该点与所属族的质心的平方误差
#         if self.initCent == 'random':
#             self.centroids = self._randCent(X, self.n_clusters)
#
#         clusterChanged = True
#         for _ in range(self.max_iter):
#             clusterChanged = False
#             for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
#                 minDist = np.inf;
#                 minIndex = -1
#                 for j in range(self.n_clusters):
#                     distJI = self._distEclud(self.centroids[j, :], X[i, :])
#                     if distJI < minDist:
#                         minDist = distJI;
#                         minIndex = j
#                 if self.clusterAssment[i, 0] != minIndex:
#                     clusterChanged = True
#                     self.clusterAssment[i, :] = minIndex, minDist ** 2
#
#             if not clusterChanged:  # 若所有样本点所属的族都不改变,则已收敛，结束迭代
#                 break
#             for i in range(self.n_clusters):  # 更新质心，即将每个族中的点的均值作为质心
#                 ptsInClust = X[np.nonzero(self.clusterAssment[:, 0] == i)[0]]  # 取出属于第i个族的所有点
#                 self.centroids[i, :] = np.mean(ptsInClust, axis=0)
#
#         self.labels = self.clusterAssment[:, 0]
#         self.sse = sum(self.clusterAssment[:, 1])
#
#     def predict(self, X):  # 根据聚类结果，预测新输入数据所属的族
#         # 类型检查
#         if not isinstance(X, np.ndarray):
#             try:
#                 X = np.asarray(X)
#             except:
#                 raise TypeError("numpy.ndarray required for X")
#
#         m = X.shape[0]  # m代表样本数量
#         preds = np.empty((m,))
#         for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
#             minDist = np.inf
#             for j in range(self.n_clusters):
#                 distJI = self._distEclud(self.centroids[j, :], X[i, :])
#                 if distJI < minDist:
#                     minDist = distJI
#                     preds[i] = j
#         return preds





# import scipy.io as sio
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# % matplotlib inline


# mat1 = '4a.mat' #这是存放数据点的文件，需要它才可以画出来。上面有下载地址
# data = sio.loadmat(mat1)
# m = data['data']
# print(m)
# x,y,z = m[0],m[1],m[2]
# ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程



# 将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x[:1000],y[:1000],z[:1000],c='y') #绘制数据点
# ax.scatter(x[1000:4000],y[1000:4000],z[1000:4000],c='r')
# ax.scatter(x[4000:],y[4000:],z[4000:],c='g')


# ax.set_zlabel('Z') #坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()




# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
#
#
#
# def read_tfrecord(filename,tensor=[1,784],num=5000):
#     filename_queue = tf.train.string_input_producer([filename]) #读入流中
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_val' : tf.FixedLenFeature(tensor, tf.float32),
#                                        })  #取出包含image和label的feature对象
#     image = tf.cast(features['img_val'], tf.float64)
#     label = tf.cast(features['label'], tf.int32)
#     images=[];labels=[]
#     with tf.Session() as sess:
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
#         coord=tf.train.Coordinator()
#         threads= tf.train.start_queue_runners(coord=coord)
#         for i in range(num):
#             example, l = sess.run([image, label])
#             images.append(np.array(example[0]))
#             tem=np.zeros((1,10))
#             tem[0][l]=1.0
#             labels.append(tem[0])
#             del tem
#         coord.request_stop()
#         coord.join(threads=threads)
#     images=np.array(images)
#     labels=np.array(labels)
#     return images,labels
#
#
# if __name__=="__main__":
#     img,lab=read_tfrecord('train.tfrecords',[1,784],100)
#     # X_embedded = TSNE(n_components=3, learning_rate=500.0, early_exaggeration=50.0, n_iter=500, init="pca",
#     #                   method="exact").fit_transform(img)
#     X_embedded = TSNE(n_components=3).fit_transform(img)
#     print("计算完成,开始绘图")
#     ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
#     lab0=[];lab1=[];lab2=[];lab3=[];lab4=[];lab5=[];lab6=[];lab7=[];lab8=[];lab9=[];i=0
#     for li in X_embedded:
#         temp=np.argmax(lab[i])
#         if temp==0:
#             lab0.append(li)
#         elif temp == 1:
#             lab1.append(li)
#         elif temp == 2:
#             lab2.append(li)
#         elif temp == 3:
#             lab3.append(li)
#         elif temp == 4:
#             lab4.append(li)
#         elif temp == 5:
#             lab5.append(li)
#         elif temp == 6:
#             lab6.append(li)
#         elif temp == 7:
#             lab7.append(li)
#         elif temp == 8:
#             lab8.append(li)
#         elif temp == 9:
#             lab9.append(li)
#         i+=1
#     labs=[lab0,lab1,lab2,lab3,lab4,lab5,lab6,lab7,lab8,lab9]
#     colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
#     i=0
#     b=[]
#     for li in labs:
#         p=[];q=[];r=[]
#         for x, y, z in li:
#             p.append(x);q.append(y);r.append(z)
#         a=ax.scatter(p, q, r, c=colors[i])  # 绘制数据点
#         b.append(a)
#         i+=1
#     ax.legend(tuple(b),('Label	Description','T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'),loc=0)
#     ax.set_zlabel('Z') #坐标轴
#     ax.set_ylabel('Y')
#     ax.set_xlabel('X')
#     plt.show()



# from sklearn import datasets
# from sklearn import decomposition
# import matplotlib.pyplot as plt
# import numpy as np
# # import seaborn
# from mpl_toolkits.mplot3d import Axes3D
# # %matplotlib notebook
#
# mnist = datasets.load_digits()
# X = mnist.data
# y = mnist.target
# pca = decomposition.PCA(n_components=3)
# new_X = pca.fit_transform(X)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.spectral)
# plt.show()


# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('./data/', one_hot=True)
# import tensorflow as tf
#
# sess = tf.InteractiveSession()
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# sess.run(tf.global_variables_initializer())
# y = tf.matmul(x, W) + b
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# for _ in range(1000):
#     batch = mnist.train.next_batch(100)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')
#
#
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# x_image = tf.reshape(x, [-1, 28, 28, 1])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(20000):
#         batch = mnist.train.next_batch(50)
#         if i % 100 == 0:
#             train_accuracy = accuracy.eval(feed_dict={
#                 x: batch[0], y_: batch[1], keep_prob: 1.0})
#             print('step %d, training accuracy %g' % (i, train_accuracy))
#         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#
#     print('test accuracy %g' % accuracy.eval(feed_dict={
#         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



# coding=utf-8
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("./data/", one_hot=True)
#
# import tensorflow as tf
# # 定义网络超参数
# learning_rate = 0.001
# training_iters = 200000
# batch_size = 64
# display_step = 20
#
# # 定义网络参数
# n_input = 784 # 输入的维度
# n_classes = 10 # 标签的维度
# dropout = 0.75 # Dropout 的概率
#
# # 占位符输入
# x = tf.placeholder(tf.float32, [None, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
# keep_prob = tf.placeholder(tf.float32)
#
# # 卷积操作
# def conv2d(name, l_input, w, b):
#     return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)
#
# # 最大下采样操作
# def max_pool(name, l_input, k):
#     return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
#
# # 归一化操作
# def norm(name, l_input, lsize=4):
#     return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
#
# # 存储所有的网络参数
# weights = {
#     'wc1': tf.Variable(tf.random_normal([11, 11, 1, 64])),
#     'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
#     'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
#     'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
#     'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
#     'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
#     'wd2': tf.Variable(tf.random_normal([4096, 4096])),
#     'out': tf.Variable(tf.random_normal([4096, 10]))
# }
# biases = {
#     'bc1': tf.Variable(tf.random_normal([64])),
#     'bc2': tf.Variable(tf.random_normal([192])),
#     'bc3': tf.Variable(tf.random_normal([384])),
#     'bc4': tf.Variable(tf.random_normal([384])),
#     'bc5': tf.Variable(tf.random_normal([256])),
#     'bd1': tf.Variable(tf.random_normal([4096])),
#     'bd2': tf.Variable(tf.random_normal([4096])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
#
# # 定义整个网络
# def alex_net(_X, _weights, _biases, _dropout):
#     # 向量转为矩阵
#     _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
#
#     # 第一层卷积
#     # 卷积
#     conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
#     # 下采样
#     pool1 = max_pool('pool1', conv1, k=2)
#     # 归一化
#     norm1 = norm('norm1', pool1, lsize=4)
#
#     # 第二层卷积
#     # 卷积
#     conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
#     # 下采样
#     pool2 = max_pool('pool2', conv2, k=2)
#     # 归一化
#     norm2 = norm('norm2', pool2, lsize=4)
#
#     # 第三层卷积
#     # 卷积
#     conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
#     # 归一化
#     norm3 = norm('norm3', conv3, lsize=4)
#
#     # 第四层卷积
#     # 卷积
#     conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
#     # 归一化
#     norm4 = norm('norm4', conv4, lsize=4)
#
#     # 第五层卷积
#     # 卷积
#     conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
#     # 下采样
#     pool5 = max_pool('pool5', conv5, k=2)
#     # 归一化
#     norm5 = norm('norm5', pool5, lsize=4)
#
#     # 全连接层1，先把特征图转为向量
#     dense1 = tf.reshape(norm5, [-1, _weights['wd1'].get_shape().as_list()[0]])
#     dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
#     dense1 = tf.nn.dropout(dense1, _dropout)
#
#     # 全连接层2
#     dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])
#     dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
#     dense2 = tf.nn.dropout(dense2, _dropout)
#
#     # 网络输出层
#     out = tf.matmul(dense2, _weights['out']) + _biases['out']
#     return out
#
# # 构建模型
# pred = alex_net(x, weights, biases, keep_prob)
#
# # 定义损失函数和学习步骤
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # 测试网络
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # 初始化所有的共享变量
# init = tf.initialize_all_variables()
#
# # 开启一个训练
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         # 获取批数据
#         sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
#         if step % display_step == 0:
#             # 计算精度
#             acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
#             # 计算损失值
#             loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
#             print ("Iter " + str(step*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
#         step += 1
#     print ("Optimization Finished!")
#     # 计算测试精度
#     print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))



# import tensorflow as tf
# import numpy as np
#
# filename_queue = tf.train.string_input_producer(["train.tfrecords"])  # 讀入數據流
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
# features = tf.parse_single_example(serialized_example,
#                                    features={
#                                        'label': tf.FixedLenFeature([], tf.int64),
#                                        'img_val': tf.FixedLenFeature([28, 28], tf.float32),
#                                    })  # 取出包含image和label的feature对象
# image = tf.cast(features['img_val'], tf.float64)
# label = tf.cast(features['label'], tf.int32)
#
# # print(image)
# # print(label)
#
# import matplotlib.pyplot as plt
# # % matplotlib inline
# from PIL import Image
#
#
# with tf.Session() as sess:  # 開始一個對話
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(4):
#         example, l = sess.run([image, label]) # 在會話中取出image和label數據
#         example *= 255
#         # print(example)
#         # example=tf.reshape(example,[28,28])
#         img = Image.fromarray(example)
#         img = img.resize([224, 224])
#         print(np.array(img))
#         plt.title("The picture number is " + str(l))
#         plt.imshow(img)
#         plt.show(img)
#         # example = np.array(example, dtype='uint8')
#         # img=Image.fromarray(example)
#         # img.thumbnail((224, 224))
#         # img.save(str(i)+'_''Label_'+str(l)+'.bmp','bmp')# 保存圖
#     # print(example, l)
#     coord.request_stop()
#     coord.join(threads)


# import keras
# import tensorflow as tf
# from PIL import Image
# from keras.models import model_from_json,load_model
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# import numpy as np
#
#
# def create_model():
#     """返回一個已創建好的 resnet50 model"""
#     model = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
#                                                  input_tensor=None, input_shape=(224,224,3),
#                                                  pooling='max',
#                                                  classes=10)
#
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# def read_tfrecord(filename, tensor=[1, 784], num=5000):
#     """
#     讀取tfrecord文件數據
#     :param filename: 文件名
#     :param tensor: 緯度
#     :param num: 讀取數據數量
#     :return: images 圖片數據列表    labels 標籤列表
#     """
#     filename_queue = tf.train.string_input_producer([filename])
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_val': tf.FixedLenFeature(tensor, tf.float32),
#                                        })
#     image = tf.cast(features['img_val'], tf.float64)
#     label = tf.cast(features['label'], tf.int32)
#     images = []
#     labels = []
#     maxlabels = []
#     with tf.Session() as sess:
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         for i in range(num):
#             example, l = sess.run([image, label])
#             images.append(np.array(example[0]))
#             tem = np.zeros((1, 10))
#             tem[0][l] = 1.0
#             labels.append(tem[0])
#             maxlabels.append([l])
#             del tem
#         coord.request_stop()
#         coord.join(threads=threads)
#     images = np.array(images)
#     labels = np.array(labels)
#     return images, labels, maxlabels
#
#
# def main():
#     images, labels, max_labels = read_tfrecord('train.tfrecords', num=1000)
#     timages, tlabels, _ = read_tfrecord('test.tfrecords', num=1000)
#     trimages = [];tsimages = []
#     for img in images:
#         img = Image.fromarray(img.reshape([28, 28])).resize([224, 224])
#         img = np.asarray(img)
#         im=[]
#         for a in img:
#             row=[]
#             for b in a:
#                 row.append([b,b+0.0,b+0.0])
#             im.append(row)
#         trimages.append(im)
#     # print(len(trimages))
#     # print(labels)
#
#     for img in timages:
#         img = Image.fromarray(img.reshape([28, 28])).resize([224, 224])
#         img = np.asarray(img)
#         im=[]
#         for a in img:
#             row=[]
#             for b in a:
#                 row.append([b,b+0.0,b+0.0])
#             im.append(row)
#         tsimages.append(im)
#
#     model=create_model()
#     model.fit(trimages,labels,batch_size=10,epochs=6)
#     model.evaluate(tsimages,tlabels,batch_size=5)
#
#     model.save('./resnet50.h5')
#
#
# def test():
#     import numpy as np
#     data = np.random.random((1,224, 224,3))
#     print(data[0][0])
#     print(len(data[0][0][0]))
#     labels = np.random.randint(2, size=(1, 10))
#     print(labels)
#     # model = create_model()
#     # model.fit(data, labels, epochs=10, batch_size=32)
#     # model.save('./resnet50.h5')
#
#
# def load():
#     timages, tlabels, _ = read_tfrecord('test.tfrecords', num=2)
#     tsimages = []
#     for img in timages:
#         img = Image.fromarray(img.reshape([28, 28])).resize([224, 224])
#         img = np.asarray(img)
#         im=[]
#         for a in img:
#             row=[]
#             for b in a:
#                 row.append([b,b+0.0,b+0.0])
#             im.append(row)
#         tsimages.append(im)
#
#     model=load_model('./resnet50.h5')
#     pre=model.predict(tsimages)
#     print(tlabels)
#     # pre = model.predict_classes(tsimages)  # 预测
#     print(pre)
#
#
# if __name__ == '__main__':
#     main()








import keras
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Input
from keras.layers import AveragePooling2D, Flatten, GlobalMaxPooling2D
from keras import layers
from keras.models import Model


def conv_block(input_tensor, bn_axis, filters, phase, name, strides=(1, 1)):
    """
    Conv2D 塊，雙路雙卷積計算
    :param input_tensor:(tensor) 輸入張量
    :param filters:(tuple) 卷積核打包
    :param strides:(int) 卷積步長
    :param BN_axis:(int) 規範化卷積軸
    :return: model
    """
    filters1, filters2, filters3 = filters  # 解包卷積核數量
    Conv_base_name = 'Conv_' + name + '_' + str(phase) + '_phase_'
    BN_base_name = 'BN_' + name + '_' + str(phase) + '_phase_'
    x = Conv2D(
        filters=filters1, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2a')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters2, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters3, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2c')(x)
    x = Activation(activation='relu')(x)

    y = Conv2D(filters3, (1, 1), strides=strides, name=Conv_base_name + '1a')(input_tensor)
    y = BatchNormalization(axis=bn_axis, name=BN_base_name + '1b')(y)

    x = layers.add([x, y])
    a = Activation('relu')(x)

    return a


def identity_block(input_tensor, bn_axis, filters, phase, name, strides=(1, 1)):
    """
    Conv2D 塊，雙路單卷積計算
    :param input_tensor:(tensor) 輸入張量
    :param filters:(tuple) 卷積核打包
    :param strides:(int) 卷積步長
    :param BN_axis:(int) 規範化卷積軸
    :return: model
    """
    filters1, filters2, filters3 = filters  # 解包卷積核數量
    Conv_base_name = 'Conv_' + name + '_' + str(phase) + '_phase_'
    BN_base_name = 'BN_' + name + '_' + str(phase) + '_phase_'
    x = Conv2D(
        filters=filters1, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2a')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters2, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters3, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2c')(x)
    x = Activation(activation='relu')(x)

    x = layers.add([x, input_tensor])
    a = Activation('relu')(x)

    return a


def my_resnet():
    inputs = Input(shape=(1, 28, 28))

    x = Conv2D(
        filters=4, kernel_size=(3, 3), padding='same', name='Conv1', data_format='channels_first')(inputs)
    x = BatchNormalization(axis=1, name='BN_Conv1')(x)
    x = Activation('relu')(x)
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first')(x)

    x = conv_block(input_tensor=x, bn_axis=1, filters=(4, 4, 64), phase=2, name='a')
    x = identity_block(input_tensor=x, bn_axis=1, filters=(4, 4, 64), phase=2, name='b')
    x = identity_block(input_tensor=x, bn_axis=1, filters=(4, 4, 64), phase=2, name='c')

    x = AveragePooling2D((2, 2), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax', name='softmax10')(x)
    # x = GlobalMaxPooling2D()(x)

    model = Model(inputs, x, name='My_Resnet')
    return model


def create_model():
    """返回一個已創建好的 resnet50 model"""
    # model = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
    #                                              input_tensor=None, input_shape=(224, 224, 3),
    #                                              pooling='max',
    #                                              classes=10)
    model = my_resnet()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def read_tfrecord(filename, tensor=[1, 784], num=5000):
    """
    讀取tfrecord文件數據
    :param filename: 文件名
    :param tensor: 緯度
    :param num: 讀取數據數量
    :return: images 圖片數據列表    labels(numpy) 標籤列表    maxlabels(int) 標籤所在項
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_val': tf.FixedLenFeature(tensor, tf.float32),
                                       })
    image = tf.cast(features['img_val'], tf.float64)
    label = tf.cast(features['label'], tf.int32)
    images = []
    labels = []
    maxlabels = []
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num):
            example, l = sess.run([image, label])
            arr=np.array(example[0])
            arr=arr.reshape((28,28))
            images.append([arr])
            tem = np.zeros((1, 10))
            tem[0][l] = 1.0
            labels.append(tem[0])
            maxlabels.append([l])
            del tem
        coord.request_stop()
        coord.join(threads=threads)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, maxlabels


def trun(arr):
    out_arr = []
    for img in arr:
        img = Image.fromarray(img.reshape([28, 28])).resize([224, 224])
        img = np.asarray(img)
        im = []
        for a in img:
            row = []
            for b in a:
                row.append([b, b + 0.0, b + 0.0])
            im.append(row)
            del row
        out_arr.append(im)
        del im
    return out_arr


def getdata(train=55000, test=5000):
    trimages, trlabels, _ = read_tfrecord('train.tfrecords', num=train,)
    teimages, telabels, _ = read_tfrecord('test.tfrecords', num=test,)
    # trimages = trun(trimages)
    # teimages = trun(teimages)

    return trimages, trlabels, teimages, telabels


def main(train=55000, test=5000, batch_size=50, epochs=20):
    trimages, trlabels, tsimages, tslabels = getdata(train=train, test=test)
    print(trlabels)
    model = create_model()
    from keras.utils import plot_model
    plot_model(model, to_file='./Resnet_model.png')
    model.fit(trimages, trlabels, batch_size=batch_size, epochs=epochs, )
    model.evaluate(tsimages, tslabels, batch_size=batch_size)
    model.save('./resnet50.h5')


def test_model(num=1000.0):
    timages, tlabels, _ = read_tfrecord('test.tfrecords', num=int(num))

    model = load_model('./resnet50.h5')
    pre = model.predict(timages)
    tr = 0
    for a, b in zip(tlabels,pre):
        a=np.argmax(a)
        b=np.argmax(b)
        if a == b:
            tr += 1
    ls = tr / num
    print("正確率為：{0}".format(ls))


def train_model():
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets('./data/', validation_size=5000)
    model=load_model('./resnet.h5')
    model.fit()



if __name__ == '__main__':
    main(epochs=30)
    # main(train=2000, test=20,epochs=5)
    # test_model(num=500)
    # x_train = np.random.random((2,5,5))
    # print(x_train)
