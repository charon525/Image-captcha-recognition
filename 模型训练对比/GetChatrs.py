import os

characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'l', 'm', 'n', 'q', 'r', 't',  'y',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8,
              'a': 9, 'b': 10, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 16, 'j': 17, 'l': 18, 'm': 19, 'n': 20, 'q': 21, 'r': 22, 't': 23, 'y': 24,
              'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31, 'H': 32, 'I': 33, 'J': 34, 'K': 35, 'L': 36, 'M': 37,'N': 38, 'O': 39, 'P': 40, 'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50}
index_char = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'a', 10: 'b', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i', 17: 'j', 18: 'l', 19: 'm', 20: 'n', 21: 'q', 22: 'r', 23: 't', 24: 'y',
            25: 'A', 26: 'B', 27: 'C', 28: 'D', 29: 'E', 30: 'F', 31: 'G', 32: 'H', 33: 'I', 34: 'J', 35: 'K', 36: 'L', 37: 'M', 38: 'N', 39: 'O', 40: 'P', 41: 'Q', 42: 'R', 43: 'S', 44: 'T', 45: 'U', 46: 'V', 47: 'W', 48: 'X', 49: 'Y', 50: 'Z'}

import pytesseract
from PIL import Image
import pygal

### tesseract进行预测
path = './train'
Num = 0
TrueNum = 0
for dir in os.listdir(path):
    path1 = os.path.join(path,dir)
    for img in os.listdir(path1):
        Num += 1
        image = Image.open(os.path.join(path1, img))
        text = pytesseract.image_to_string(image)
        if dir[0] != text:
            print("预测：{}， 实际：{}".format(text, dir[0]))
        else:
            TrueNum += 1
print("acc: %.6f" %(TrueNum / Num))

# 数据来自实验过程，可视化是在windows下实现的
chars = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'l', 'm', 'n', 'q', 'r', 't',  'y']
labelacc = [0.51, 0.74, 0.74, 0.51, 0.68, 0.52, 0.38, 0.26, 0.66, 0.28, 0.54, 0.91, 0.35, 0.81,
             0.64, 0.65, 0.57, 0.53, 0.63, 0.61, 0.73, 0.54, 0.53, 0.58, 0.6, 0.56, 0.34, 0.63, 0.17,
             0.61, 0.11, 0.61, 0.44, 0.05, 0.68, 0.66, 0.62, 0.73, 0.62, 0.53, 0.64, 0.68, 0.45, 0.61, 0.39, 0.89, 0.68, 0.55, 0.76, 0.71, 0.41]
labelnum = [633, 596, 614, 625, 642, 641, 642, 667, 610, 595, 656, 611, 632, 641, 620, 628, 643, 619,
            658, 622, 641, 670, 630, 610, 628, 630, 701, 579, 614, 594, 647, 604, 608, 638, 617, 653,
            678, 647, 563, 594, 655, 599, 632, 608, 651, 582, 609, 599, 629, 658, 597]
modelAndAcc = [['tesseract OCR', 0.01], ['matchTemplate', 0.54], ['LogisticRegression', 0.82],
               ['DecionTree', 0.42], ['RandomForeast', 0.86], ['BPNN', 0.88], ['CNN', 0.96]]

bar_chart1_acc = pygal.HorizontalBar()
bar_chart2_acc = pygal.HorizontalBar()
bar_chart1_acc.title = 'Numbers & LowercaseLetters Accuracy'
bar_chart2_acc.title = 'UppercaseLetters Accuracy'
for i in range(len(chars)):
    if ord(chars[i][0]) >= 65 and ord(chars[i][0]) <= 90:
        bar_chart2_acc.add(chars[i], labelacc[i])
    else:
        bar_chart1_acc.add(chars[i], labelacc[i])
bar_chart1_acc.render_to_png('./charts/Numbers & LowercaseLetters_acc.png')
bar_chart2_acc.render_to_png('./charts/UppercaseLetters_acc.png')


bar_chart1_num = pygal.HorizontalBar()
bar_chart2_num = pygal.HorizontalBar()
bar_chart1_num.title = 'Numbers & LowercaseLetters Num'
bar_chart2_num.title = 'UppercaseLetters Num'
for i in range(len(chars)):
    if ord(chars[i][0]) >= 65 and ord(chars[i][0]) <= 90:
        bar_chart2_num.add(chars[i], labelnum[i])
    else:
        bar_chart1_num.add(chars[i], labelnum[i])
bar_chart1_num.render_to_png('./charts/Numbers & LowercaseLetters_num.png')
bar_chart2_num.render_to_png('./charts/UppercaseLetters_num.png')


bar_chart3 = pygal.HorizontalBar()
bar_chart3.title = 'Model Accuracy'
for i in modelAndAcc:
    bar_chart3.add(i[0], i[1])
bar_chart3.render_to_png('./charts/Model Accuracy.png')