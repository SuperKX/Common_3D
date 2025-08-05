'''
    图片处理，公共功能
    STD_COLORS      定义的标准颜色，可以通过字符串索引
    std_color_idx_search    根据索引返回颜色
    init_color      指定一些变量的颜色
    draw_polygon    在图片上绘制多边形
    draw_rect       绘制包围盒
    pic_resize      图片缩放
'''
import cv2
import random
import numpy as np

# 全局定义颜色
STD_COLORS = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "random": (int(random.random() * 256), int(random.random() * 256), int(random.random() * 256))  # 随机
}

def std_color_idx_search(idx):
    '''
    输入任意索引（需要合格），返回对应的标准颜色。
    注意：
        1）如果超出标准颜色数量，则返回任意颜色
    :param idx: 索引下标
    :return: 颜色元组，如(255, 0, 255)
    '''
    if not isinstance(idx,int):
        raise ValueError(f"颜色索引错误")
    if idx<0:
        raise ValueError(f"颜色索引为负")

    items_list = list(STD_COLORS.items())
    num = len(items_list)  # 总数
    if idx > num:
        idx = num-1
    key, value = items_list[idx]
    return value


def init_color(*args):
    '''
    颜色赋值写出 如:  color_win, color_door = init_color("red","green")
    说明：
         1）支持字符串、三元组作为输入，支持随机颜色
    :param args: 输入颜色名
    :return:
    '''
    tuple_out = []
    for colori in args:
        if isinstance(colori, str):
            if colori not in STD_COLORS:
                raise ValueError(f"没有定义颜色: {colori}")
            tuple_out.append(STD_COLORS[colori])
        elif isinstance(colori, tuple) and len(colori) == 3:  # 三元组
            tuple_out.append(colori)
        else:
            raise ValueError(f"颜色格式不合法: {colori}")
    return tuple(tuple_out)

def draw_polygon(img, polygon, color, thickness=5):
    '''
    绘制多边形在cv2的图片上，
    :param img:  图片cv2
    :param polygon: polygon是可迭代的列表，如二维list[]
    :param color: 颜色，三元组，如(0, 255, 255)，或者全局定义的颜色
    :param thickness:   线宽度
    :return: img 返回绘制后
    '''
    num = len(polygon)
    for i in range(num):
        point1 = polygon[i]  # 相邻两个点
        point1 = list(map(int, point1))
        point2 = polygon[(i + 1) % num]
        point2 = list(map(int, point2))
        cv2.line(img, point1, point2, color, thickness)
    return img

def draw_rect(img, rect, color, thickness=5):
    '''
    绘制bbox在cv2的图片上，
    注意：
        1）输入的rect非像素，会自动纠正
    :param img:  图片cv2
    :param rect: rect是 方框[x_min,y_min,x_max,y_max]
    :param color: 颜色，三元组，如(0, 255, 255)，或者全局定义的颜色
    :param thickness:   线宽度
    :return: img 返回绘制后
    '''
    rect = list(map(int, rect))
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness)
    return img

def draw_pixs(img, pixs, color, alpha=0.3):
    '''
    将像素掩码会知道图片img上
    :param img: cv2读入图片
    :param pixs: 像素掩码（bool）
    :param color: 填充颜色
    :param alpha: 透明度
    :return:
    '''
    # 确保pixs是布尔类型
    pixs = pixs.astype(bool)
    # 确保颜色是元组
    if not isinstance(color, tuple):
        raise TypeError("color must be a tuple (B, G, R)")
    # 确保颜色是 BGR 格式的元组
    if len(color) != 3:
        raise ValueError("color must be a BGR tuple with 3 elements")
    # 创建一个与图片大小相同的颜色图层
    color_layer = np.zeros_like(img, dtype=np.uint8)
    color_layer[:] = color  # 将颜色图层填充为指定颜色
    # 使用像素掩码选择要修改的区域
    img[pixs] = cv2.addWeighted(img, 1 - alpha, color_layer, alpha, 0)[pixs]
    return img


def draw_face(img, faces, color, alpha=0.3):
    '''
    图片绘制多个面片
    :param img: 图片
    :param faces: 面片列表 list 三维，多个面[[[边],面]，多个面]
    :param color: 颜色
    :param alpha:
    :return:
    '''
    overlay = img.copy()  # 创建图像副本，避免直接修改原始图像
    faces = [np.array(face) for face in faces]
    cv2.fillPoly(overlay, faces, color)  # 在副本上绘制多边形
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # 将副本与原始图像混合，实现透明效果
    return img

def draw_str(img, text_list, pos_list, color, thickness=5):
    '''
    绘制多段文字
    :param img: 图片
    :param text_list: 文本内容列表
    :param pos_list: 位置
    :param color: 颜色
    :param thickness: 粗细
    :return:
    '''
    for i,text in enumerate(text_list):
        text = str(int(text*100))
        pos = tuple(pos_list[i])
        pos = list(map(int, pos))
        cv2.putText(img, text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color, fontScale=thickness//2, thickness=thickness, lineType=None, bottomLeftOrigin=None)


def pic_resize(img, width_max):
    '''
    图片缩放，解决过大图片写出问题
    :param img: 输入的图片
    :param width_max: 最长边大小 pix
    :return: img_resized
    '''
    pic_h,pic_w,_ = img.shape  #
    pic_h_new = int(pic_h/pic_w*width_max)
    img_resized = cv2.resize(img, (width_max, pic_h_new), interpolation=cv2.INTER_LINEAR)
    return img_resized

def mask_main_poly(bool_mask):
    '''
    获取二维掩码主要部分的外轮廓线
    bool_mask bool类型的ndarray二维。
    '''
    # 1. bool掩码转二值图 [0,255]
    mask = bool_mask.astype(np.uint8) * 255
    # 2. 去噪：中值滤波或高斯滤波
    mask = cv2.medianBlur(mask, 5)  # 去除孤立噪点
    # 3. 形态学操作修复破碎区域
    kernel = np.ones((5, 5), np.uint8)
    # 膨胀：连接邻近前景像素
    mask = cv2.dilate(mask, kernel, iterations=2)
    # 腐蚀：去除膨胀后多余的像素，恢复形状
    mask = cv2.erode(mask, kernel, iterations=2)
    # 4. 填充内部空洞(此部分代码有问题)
    # filled_mask = mask.copy()
    # cv2.floodFill(filled_mask, None, (0, 0), 0)  # 填充背景为0
    # filled_mask = cv2.bitwise_not(filled_mask)  # 取反得到填充后的前景
    # mask = cv2.bitwise_or(mask, filled_mask)  # 合并原掩码与填充结果
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 筛选有效轮廓（按面积排序，取最大轮廓）
    main_contour = []
    if contours:
        # 按轮廓面积降序排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_contour = contours[0]  # 最大轮廓作为目标外轮廓
    else:
        print("未检测到有效轮廓")
    return main_contour



