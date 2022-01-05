
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 最小二乘法
# inout:输入二维数组
def LeastSquare(input):

    # 样本数
    N = input.shape[0]

    # 计算直线斜率和截距
    temp1 = N*np.sum(input[:,0]*input[:,1])
    temp2 = np.sum(input[:,0])*np.sum(input[:,1])
    temp3 = N*np.sum(pow(input[:,0],2))
    temp4 = pow(np.sum(input[:,0]),2)
    k = (temp1-temp2)/(temp3-temp4)
    temp5 = np.sum(input[:, 1]) / N
    temp6 = np.sum(input[:, 0]) / N
    b = temp5-k*temp6

    X = np.array([k,b])

    return X

# RANSAC实现
# 数据集:input
# 随机采样数:rnum
# 迭代次数:lnum
# 筛选内点的距离阈值:lenthr
def ransac(input,rnum,lnum,lenthr):

    # 样本数
    totnum = input.shape[0]

    # 斜率与截距
    k = 0
    b = 0

    # 最多内点数
    internum = rnum

    # 循环
    for i in range(lnum):

        # 随机采样点索引
        # rs = np.random.sample(range(0,totnum-1),rnum)
        rs = np.random.choice(totnum,rnum,replace=False)

        # 选中数据
        select = input[rs,:]

        # 未选中数据
        non_idx = np.setdiff1d(range(0,totnum),rs)
        non = input[non_idx,:]

        # 计算最小二乘
        temp_result = LeastSquare(select)

        # 未选中点中的内点
        count = 0

        # 计算未选中数据中有多少属于内点
        for j in range(non.shape[0]):

            # 距离判断
            if k*non[j,0]+b>lenthr:
                count = count+1

        # 若当前迭代得到的内点总数比之前多,则更新.
        if rnum+count>=internum:
            internum = rnum+count
            k = temp_result[0]
            b = temp_result[1]
            count = 0

    X = np.array([k,b])
    return X

# 主函数
if __name__ == '__main__':

    # 数据
    # data = np.array([[1,2],[2,5],[3,10],[4,11],[5,14],[6,17],[7,20],[8,23],[9,26],[10,29]])
    # data = np.array([[1, 2.1], [2, 4.8], [3, 10], [3.9, 11], [5, 14], [6.2, 17.1], [7, 19.4], [8, 23.3], [9.1, 26], [10, 29]])
    data = np.array([[1, 5.1], [2, 4.8], [3, 10], [3.9, 11], [5, 14], [6.2, 17.1], [7, 19.4], [8, 23.3], [9.1, 26], [10, 29]])

    # 调用ransac
    Res = ransac(data,3,100,0.6)

    print(Res)

