import os
import cv2
characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'l', 'm', 'n', 'q', 'r', 't',  'y',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 
              'a': 9, 'b': 10, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 16, 'j': 17, 'l': 18, 'm': 19, 'n': 20, 'q': 21, 'r': 22, 't': 23, 'y': 24, 
              'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31, 'H': 32, 'I': 33, 'J': 34, 'K': 35, 'L': 36, 'M': 37,'N': 38, 'O': 39, 'P': 40, 'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50}
index_char = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'a', 10: 'b', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i', 17: 'j', 18: 'l', 19: 'm', 20: 'n', 21: 'q', 22: 'r', 23: 't', 24: 'y', 
            25: 'A', 26: 'B', 27: 'C', 28: 'D', 29: 'E', 30: 'F', 31: 'G', 32: 'H', 33: 'I', 34: 'J', 35: 'K', 36: 'L', 37: 'M', 38: 'N', 39: 'O', 40: 'P', 41: 'Q', 42: 'R', 43: 'S', 44: 'T', 45: 'U', 46: 'V', 47: 'W', 48: 'X', 49: 'Y', 50: 'Z'}
    #            数字             英文字母（大小写）


def matchValue(img):
    values = [0] *51
    path = './template'
    for dir in sorted(os.listdir(path)):
        dir_values = [] 
        path1 = (os.path.join(path, dir))
        for num, filename in enumerate(sorted(os.listdir(path1))):
            path2 = os.path.join(path1, filename)
            imgtemplate = cv2.imread(path2)
            img = cv2.resize(img, (25, 60))
            result = cv2.matchTemplate(img, imgtemplate, cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(result)
            dir_values.append(score)
        values[char_index[dir[0]]] = max(dir_values)
    return index_char[values.index(max(values))], max(values)


if __name__ == "__main__":
    path = './train'
    totalNum, trueNum = 0, 0
    for directory in sorted(os.listdir(path)):
        dirnum = 0
        path1 = os.path.join(path, directory)
        for filename in sorted(os.listdir(path1)):
            img = cv2.imread(os.path.join(path1, filename))
            predict, value = matchValue(img)
            if predict == str(directory[0]):
                trueNum += 1
                dirnum += 1
            else:
                print("实际值：{},预测值: {}, 最大匹配值：{}".format(str(directory[0]), predict, value))
            totalNum += 1
        print("字符：{}， 正确数量： {}， 总数：{}， acc:{}".format(directory[0], dirnum, len(os.listdir(path1)), dirnum/len(os.listdir(path1))))
    print("accuracy: %.6f" %(trueNum/totalNum*1.0))