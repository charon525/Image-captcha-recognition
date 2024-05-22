import cv2
import os
import numpy as np
characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'l', 'm', 'n', 'q', 'r', 't',  'y',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 
              'a': 9, 'b': 10, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 16, 'j': 17, 'l': 18, 'm': 19, 'n': 20, 'q': 21, 'r': 22, 't': 23, 'y': 24, 
              'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31, 'H': 32, 'I': 33, 'J': 34, 'K': 35, 'L': 36, 'M': 37,'N': 38, 'O': 39, 'P': 40, 'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50}
index_char = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'a', 10: 'b', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i', 17: 'j', 18: 'l', 19: 'm', 20: 'n', 21: 'q', 22: 'r', 23: 't', 24: 'y', 
            25: 'A', 26: 'B', 27: 'C', 28: 'D', 29: 'E', 30: 'F', 31: 'G', 32: 'H', 33: 'I', 34: 'J', 35: 'K', 36: 'L', 37: 'M', 38: 'N', 39: 'O', 40: 'P', 41: 'Q', 42: 'R', 43: 'S', 44: 'T', 45: 'U', 46: 'V', 47: 'W', 48: 'X', 49: 'Y', 50: 'Z'}


def cv_show(img, name):
    '''
    显示图片
    :param img: 要显示的图片
    :param name: 图片名称
    :return: None
    '''
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RemoveSmallCC(img, small, connectivity=8):
    '''
    去除小连通域，即缓解图片上的斑点和字符粘连问题。
    :param img: 待处理的图片
    :param small: 连通域上限
    :param connectivity: 连通域构成方式为8邻域或者4邻域
    :return: 处理后的图片。
    '''
    # (1)获取连通域标签
    ret, labels = cv2.connectedComponents(img, connectivity=connectivity)
    # (2)去小连通域
    for n in range(ret + 1):  # 0 ~ max
        num = 0  # 清零
        for elem in labels.flat:
            if elem == n:
                num += 1
        if num < small:  # 去除小连通域
            img[np.where(labels == n)] = 0
    return img


def projection(img):
    '''
    将图片每一列的像素值累加起来实现将图片从二维投影到一维
    :param img: 待处理的图片
    :return: 投影后的以为数组
    '''
    img_tmp = img / 255.0
    proj = np.zeros(shape=(img_tmp.shape[1]))
    for col in range(img_tmp.shape[1]):
        v = img_tmp[:, col]
        proj[col] = np.sum(v)
    return proj


def Removeline(img):
    '''
    去除图片上的干扰线
    :param img: 待处理图片
    :return: 处理后的图片
    '''
    proj = projection(img)
    flaw_area = np.where(proj < 5)
    img[:, flaw_area] = 0  # 去除横线
    return RemoveSmallCC(img, 200)


def Extract_num(proj):
    '''
    根据投影后的数组，判别4位字符的交杂情况，有：
    1+1+1+1型、1+1+2型、2+2型、1+3型、4型
    :param proj: 投影后的数组
    :return: 提取的片段和每一个片段的起始位置
    '''
    num = 0
    COUNT, LOC = [], []
    for i in range(len(proj)):
        if proj[i]:  # 如果非零则累加
            num += 1
            if i == 0 or proj[i-1] == 0:  # 记录片段起始位置
                start = i
            if i == len(proj) -1 or proj[i+1] == 0:  # 定义派那段结束标志，并记录片段
                end = i
                if num > 10:   # 提取有效片段
                    COUNT.append(num)
                    LOC.append((start, end))
                num = 0
    return LOC, COUNT


# 分割数字
def Segment4_Num(COUNT, LOC):
    '''
    根据提取出来的片段进行裁剪
    :param COUNT: 字符片段数目
    :param LOC: 每一个片段的起始位置
    :return:
    '''
    if len(COUNT) > 4 or len(COUNT) <= 0:
        return 
    # 数字部分分析
    if len(COUNT) == 4:  # (1,1,1,1)
        return LOC

    if len(COUNT) == 3:  # (1,1,2)
        idx = np.argmax(np.array(COUNT))  # 最大片段下标
        r = LOC[idx]  # 最大片段位置
        start = r[0]
        end = r[1]
        m = (r[0] + r[1]) // 2  # 中间位置
        # 修改LOC[idx]
        LOC[idx] = (start, m)
        LOC.insert(idx + 1, (m + 1, end))
        return LOC

    if len(COUNT) == 2:  # (2,2)or(1,3)
        skew = max(COUNT) / min(COUNT)  # 计算偏移程度
        if skew < 1.7:  # 认为是（2，2）
            start1 = LOC[0][0]
            end1 = LOC[0][1]
            start2 = LOC[1][0]
            end2 = LOC[1][1]
            m1 = (start1 + end1) // 2
            m2 = (start2 + end2) // 2
            return [(start1, m1), (m1 + 1, end1), (start2, m2), (m2 + 1, end2)]
        else:  # 认为是（1，3）
            idx = np.argmax(np.array(COUNT))  # 最大片段下标
            start = LOC[idx][0]
            end = LOC[idx][1]
            m1 = (end - start) // 3 + start
            m2 = (end - start) // 3 * 2 + start
            # 修改LOC[idx]
            LOC[idx] = (start, m1)
            LOC.insert(idx + 1, (m1 + 1, m2))
            LOC.insert(idx + 2, (m2 + 1, end))
            return LOC
    if len(COUNT) == 1:  # (4)
        start = LOC[0][0]
        end = LOC[0][1]
        m1 = (end - start) // 4 + start
        m2 = (end - start) // 4 * 2 + start
        m3 = (end - start) // 4 * 3 + start
        return [(start, m1), (m1 + 1, m2), (m2 + 1, m3), (m3 + 1, end)]


def imgStore(img, filename, train=True):
    if train:
        path = './train/' + filename + '_' +str(char_index[filename])
    else:
        path = './test/' + filename + '_' +str(char_index[filename])
    num = len(os.listdir(path))
    path_img =  path + '/' + str(num) + '.png'
    cv2.imwrite(path_img, img)

def Process(path, filename, train=True):
    img = cv2.imread(path)
    # cv_show(img, 'origin')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 首先对图像进行灰度处理
    _, img = cv2.threshold(img, int(0.9*255), 255, cv2.THRESH_BINARY_INV)  # 二值化操作

    img = RemoveSmallCC(img, 200, connectivity=4)  # 去除小连通域

    kernel_erode = np.ones((3, 3), np.uint8)  # 进行图片腐蚀操作
    img = cv2.erode(img, kernel_erode, iterations=1)

    img = RemoveSmallCC(img, 30)  # 去除小连通域

    kernel_dilate = np.ones((3, 3), np.uint8)  # 进行图片膨胀操作
    img = cv2.dilate(img, kernel_dilate, iterations=1)

    img = Removeline(img)

    proj = projection(img)
    loc, count = Extract_num(proj)

    # 优化count
    # 分割数字
    LOC = Segment4_Num(count, loc)

    NUM0 = img[:, LOC[0][0]:LOC[0][1]]
    NUM0 = RemoveSmallCC(NUM0, 50)
    NUM1 = img[:, LOC[1][0]:LOC[1][1]]
    NUM1 = RemoveSmallCC(NUM1, 50)
    NUM2 = img[:, LOC[2][0]:LOC[2][1]]
    NUM2 = RemoveSmallCC(NUM2, 50)
    NUM3 = img[:, LOC[3][0]:LOC[3][1]]
    NUM3 = RemoveSmallCC(NUM3, 50)

    # cv_show(NUM0, filename[0])

    imgStore(NUM0, filename[0],train)

    # cv_show(NUM1, filename[1])
    imgStore(NUM1, filename[1],train)

    # cv_show(NUM2, filename[2])
    imgStore(NUM2, filename[2], train)

    # cv_show(NUM3, filename[3])
    imgStore(NUM3, filename[3], train)


if __name__ == "__main__":
    path = './images'
    num = 0
    for filename in os.listdir(path):
        num += 1
        train = True
        if num < 0.85 * len(os.listdir(path)):
            train = True
        else:
            train = False
        print("{}/{}".format(num, len(os.listdir(path))))
        imagepath = os.path.join(path, filename)
        try:
            Process(imagepath, filename, train)
        except:
            continue
    print("Process ImageData Successfully!")

