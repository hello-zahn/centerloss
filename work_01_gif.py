"""
图片转化为gif会存在序号乱码问题
"""
import os
import re

import PIL.Image

img_path = 'img4'
imgs = os.listdir(img_path)
# 读取多张图片
image_list = []


# 使用自定义排序函数对文件名进行排序
def extract_number(s):
    match = re.search(r'\d+', s)  # 在文件名中查找数字
    return int(match.group()) if match is not None else -1  # 如果找到数字则返回，否则返回-1


sorted_files = sorted(imgs, key=extract_number)
for s in sorted_files:
    path = os.path.join(img_path, s)
    img = PIL.Image.open(path)
    image_list.append(img)

image_list[0].save('output_cifar100.gif', save_all=True, append_images=image_list[1:], optimize=False,
                   duration=100, loop=0)
pass
