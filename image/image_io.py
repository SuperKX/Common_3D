'''
  图片io，及相关方式获取属性
    cv_imread   读入中文路径的图片
    cv_imwrite  写出中文路径的图片
    img_size    从文件快速获取图片大小
'''
import cv2
import numpy as np
from PIL import Image


# 中文路径读入
# img = cv2.imread(path)  # 普通方法
def cv_imread(file_path):
  """
  支持中文路径的图像读取函数
  file_path 文件路径
  """
  try:
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img
  except Exception as e:
    print(f"Error reading image: {e}")
    return None

# 中文路径写出
def cv_imwrite(file_path, img):
  """
  支持中文路径的图像读取函数
  file_path 文件路径
  img 图像信息
  """
  try:
    # cv2.imwrite(file_path, img)  #普通方法
    cv2.imencode('.jpg', img)[1].tofile(file_path)
  except Exception as e:
    print(f"Error reading image: {e}")


def img_size(file_path):
  '''
  快速获取照片尺寸
  :param file_path:
  :return: pic_w, pic_h
  '''
  with Image.open(file_path) as img:
    pic_w, pic_h = img.size

  return pic_w, pic_h
