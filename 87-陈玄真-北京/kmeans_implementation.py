
'''实现k均值聚类'''

# chen x.z. 2021.12.22
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据样本
data = np.array([[2,3,1],[4,6,9],[7,1,6],[4,4,2],[1,5,3]])

# 创建与样本对应的聚类序号
xuhao = np.zeros(data.shape[0])

# k均值聚类函数
# 输入参数1: K个聚类
# 输入参数2: 输入数据
def k_means_clustering(K, input, max_iter):

    # 全局数组声明
    global xuhao
    global data

    # 获取样本个数和特征维数
    num, dims = input.shape

    # 初始聚类中心
    centers = np.random.randint(np.min(input),np.max(input),(K,3))

    # 迭代循环
    for i in range(max_iter):

        # 样本循环
        for j in range(input.shape[0]):

            # 距离列表
            distance_list = np.zeros([K, 1])

            # 当前数据样本
            tempdata = input[j,:]

            # 聚类中心循环
            for k in range(K):

                # 计算样本与聚类中心的距离
                temp_center = centers[k,:]
                temp_dist = np.linalg.norm(tempdata - temp_center)

                # 存入距离列表
                distance_list[k] = temp_dist

            # 计算最小距离
            min_dis = min(distance_list)

            # 查找最小距离索引
            index = np.where(distance_list==min_dis)
            index = index[0]

            # 修改样本归入的类别序号
            xuhao[j] = index + 1

        # 聚类中心更新
        for w in range(K):

            # w从0~(K-1),对应的类别序号是1~K.
            temp_index = np.where(xuhao==w+1)

            if np.size(temp_index) != 0:
                # 取出当前类别的所有样本数据
                clustering_data = input[temp_index,:]

                # 计算聚类中心
                temp_result = np.mean(clustering_data, axis=0)
                temp = np.mean(temp_result,axis=0)
                centers[w, :] = temp

    # 返回聚类结果
    return xuhao

# 主函数
if __name__ == '__main__':

    # 聚类数
    clustering_num = 2

    # 最大迭代次数
    iterations = 200

    # 聚类结果
    xuhao = k_means_clustering(clustering_num, data, iterations)
    print(xuhao)

    '''# 测试
    A = np.array([[3,4],[1,2],[4,5],[6,2],[3,2]])
    B = np.array([1,2,1,1,3])
    C = np.array([[1.2,2.3],[4.3,2.1]])
    idx = np.where(B == 1)
    H = A[idx,:]
    E = np.mean(H,axis=0)
    F = np.mean(E,axis=0)
    p = 0
    C[p,:] = F
    # print(C)

    A1_min = np.min(A)
    print(A1_min)
    A1_max = np.max(A)
    # print(A1_max)'''