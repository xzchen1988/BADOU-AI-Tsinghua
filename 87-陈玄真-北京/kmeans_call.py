
'''基于k-means实现图像聚类'''

# chen x.z. 2021.12.15
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
source = cv2.imread('AWACS.jpeg', cv2.IMREAD_GRAYSCALE)

# 图像拉直并将uint8型转换为float32型
data = source.reshape((source.shape[0] * source.shape[1], 1))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# K-Means聚类 聚集成3类
# 输入参数1: 原始数据
# 输入参数2: 聚类数
# 输入参数3: 预设标签
# 输入参数4: 迭代终止条件
# 输入参数5: 试验重复次数
# 输入参数6: 初始聚类中心选取
# 输出参数1: 紧密度
# 输出参数2: 数据点到相应聚类中心距离的平方
# 输出参数3: 聚类中心构成的数组
compactness, labels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 聚类结果图像
result = labels.reshape((source.shape[0], source.shape[1]))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
io.imshow(source)
ax2 = fig.add_subplot(1, 2, 2)
io.imshow(result, cmap='gray')
plt.show()