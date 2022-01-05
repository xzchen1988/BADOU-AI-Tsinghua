
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 均值哈希算法计算图像相似度
# Img:rgb图像
# height:图像调整后的高度
# width:图像调整后的宽度
# 输出:hash指纹
def MeanHash(Img,height,width):

    # 图像缩放
    Img = cv2.resize(Img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    # Img = np.resize(Img, (height, width))

    # 转换为灰度图
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    # 计算像素灰度均值
    junzhi = np.mean(gray)

    # 生成均值哈希指纹
    binary = gray >= junzhi
    binary = binary.astype(int)
    mhs = binary.reshape((1,height*width))

    # 返回哈希值
    return mhs

# 差值哈希算法计算图像相似度
# Img:rgb图像
# height:图像调整后的高度
# width:图像调整后的宽度
# 输出:hash指纹
def DiffHash(Img,height,width):

    # 图像缩放
    Img = cv2.resize(Img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    # 转换为灰度图
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    # 哈希指纹
    dhs = np.array([])

    for i in range(height):
        for j in range(width-1):
            if gray[i,j] >= gray[i,j+1]:
                dhs = np.append(dhs,1)
            else:
                dhs = np.append(dhs,0)

    return dhs

# 主函数
if __name__ == '__main__':

    # 读取图像
    src1 = cv2.imread("AWACS.jpeg")
    src2 = cv2.imread("Y-20.jpeg")

    # 调用均值哈希算法计算哈希值
    meanhs1 = MeanHash(src1, 10, 10)
    meanhs2 = MeanHash(src2, 10, 10)

    # 计算均值哈希相似度
    meanIdx = (meanhs1!=meanhs2)
    meanIdx = meanIdx.astype(int)
    meanSim = np.sum(meanIdx)

    # 调用差值哈希算法计算哈希值
    diffhs1 = DiffHash(src1,10,11)
    diffhs2 = DiffHash(src2, 10, 11)

    # 计算差值哈希相似度
    diffIdx = (diffhs1 != diffhs2)
    diffIdx = diffIdx.astype(int)
    diffSim = np.sum(diffIdx)

    a=1