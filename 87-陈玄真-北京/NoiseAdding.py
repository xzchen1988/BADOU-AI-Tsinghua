
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 添加高斯噪声
# input:灰度图像
# output:高斯噪声污染图像
# mean:高斯分布的像素值归一化均值
# var:高斯分布的像素值归一化方差
def GaussNoise(input,mean,var):

    # 获取尺寸
    height, width = input.shape

    # 将图像拉直成向量
    vec = input.reshape(1,height*width)

    # 求最大灰度
    # 注意:这里要用np.max,不能用max.
    gray_max = np.max(vec)

    # 生成高斯随机向量
    rand_vec = np.random.normal(mean,var,height*width)

    # 创建输出图像
    output = vec/gray_max + rand_vec
    output = (output - np.min(output)) / (np.max(output)-np.min(output))

    # 逆映射:返回[0,255]
    output = 255*output
    output = output.astype(np.uint8)

    # 调整尺寸
    output = output.reshape(height, width)

    return output

# 添加椒盐噪声
# 输入灰度图像
# snr:信噪比
# percent:椒噪声占椒盐噪声的比例
def SaltPepperNoise(input,snr,percent):

    # 椒噪声在椒盐噪声中占比
    height,width = input.shape

    # 椒盐噪声总像素数
    # 转换类型时要注意取值范围,noise_num大于uint8的最大值,则转换后为0.
    noise_num = np.floor(height*width*snr)
    noise_num = noise_num.astype(np.uint64)

    # 椒噪声总像素数
    pepper_num = np.floor(noise_num*percent)
    pepper_num = pepper_num.astype(np.uint64)

    # 盐噪声总像素数
    salt_num = noise_num - pepper_num

    # 正常像素总像素数
    pixel_num = height*width - noise_num

    # 生成标志向量
    pixel = -1*np.ones(pixel_num.astype(np.uint64))
    salt = -2*np.ones(salt_num.astype(np.uint64))
    pepper = -3*np.ones(pepper_num.astype(np.uint64))

    # 合成像素向量
    vec = np.concatenate((pixel,salt,pepper))
    vec = np.random.permutation(vec)

    # 创建输出图像
    output = input
    output = output.reshape(height*width,1)

    # 赋值循环
    for k in range(len(output)):
        if vec[k] == -2:
            output[k] = 255
        else:
            if vec[k] == -3:
                output[k] = 0
            else:
                continue

    # 调整尺寸
    output = output.reshape(height,width)

    # 返回加噪结果
    return output

if __name__ == '__main__':

    # 读取图像
    src = cv2.imread("AWACS.jpeg")

    # 转化为灰度图
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 添加高斯噪声后的图像
    GauNoiImg = GaussNoise(gray.copy(),0.004,0.05)

    # 添加椒盐噪声后的图像
    SalPepNoiImg = SaltPepperNoise(gray.copy(),0.03,0.6)

    # cv2.imshow("灰度", gray)
    # cv2.waitKey(2000)


    # 绘图
    fig2 = plt.figure()
    ax21 = fig2.add_subplot(1, 3, 1)
    io.imshow(gray)
    plt.title('Original Image')
    ax22 = fig2.add_subplot(1, 3, 2)
    io.imshow(GauNoiImg)
    plt.title('Gauss Noise')
    ax23 = fig2.add_subplot(1, 3, 3)
    io.imshow(SalPepNoiImg)
    plt.title('Salt and Pepper Noise')
    plt.show()