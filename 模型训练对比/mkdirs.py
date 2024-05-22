import os
characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'l', 'm', 'n', 'q', 'r', 't',  'y',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 
              'a': 9, 'b': 10, 'd': 11, 'e': 12, 'f': 13, 'g': 14, 'h': 15, 'i': 16,
              'j': 17, 'l': 18, 'm': 19, 'n': 20, 'q': 21, 'r': 22, 't': 23, 'y': 24,
              'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31, 'H': 32,
              'I': 33, 'J': 34, 'K': 35, 'L': 36, 'M': 37,'N': 38, 'O': 39, 'P': 40,
              'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50}
index_char = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'a', 10: 'b', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i',
              17: 'j', 18: 'l', 19: 'm', 20: 'n', 21: 'q', 22: 'r', 23: 't', 24: 'y',
            25: 'A', 26: 'B', 27: 'C', 28: 'D', 29: 'E', 30: 'F', 31: 'G', 32: 'H',
              33: 'I', 34: 'J', 35: 'K', 36: 'L', 37: 'M', 38: 'N', 39: 'O', 40: 'P',
              41: 'Q', 42: 'R', 43: 'S', 44: 'T', 45: 'U', 46: 'V', 47: 'W', 48: 'X', 49: 'Y', 50: 'Z'}


# 创建文件夹
if not os.path.exists('./train'):
    os.mkdir('./train')
if not os.path.exists('./test'):
    os.mkdir('./test')
if not os.path.exists('./template'):
    os.mkdir('./template')
if not os.path.exists('./src'):
    os.mkdir('./src')
if not os.path.exists('./charts'):
    os.mkdir('./charts')
if not os.path.exists('./images'):
    os.mkdir('./images')
for c in characters:
    folder_name = c + '_' +  str(char_index[c])
    if not os.path.exists('./train/' + folder_name):
        os.makedirs('./train/' + folder_name)
    if not os.path.exists('./test/' + folder_name):
        os.makedirs('./test/' + folder_name)
    if not os.path.exists('./template/' + folder_name):
        os.makedirs('./template/' + folder_name)
print("make directory successfully!")


