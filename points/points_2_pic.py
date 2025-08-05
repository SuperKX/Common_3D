import numpy as np
import cv2
import struct
from PIL import Image


def transform_point_cloud(point_cloud, projection_dir):
    # 输入检查
    projection_dir = np.array(projection_dir, dtype=np.float64).flatten()
    if projection_dir.size != 3:
        raise ValueError("投影方向必须为三维向量")

    # 步骤1：单位化投影方向的逆方向
    v = -projection_dir
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("投影方向不能为零向量")
    u = v / norm  # 新z轴正方向

    # 步骤2：构造旋转矩阵
    if np.allclose(u, [1, 0, 0]) or np.allclose(u, [-1, 0, 0]):
        # 特殊情况：u与x轴共线，改用y轴计算x轴方向
        x_dir = np.cross([0, 1, 0], u)
    else:
        x_dir = np.cross([1, 0, 0], u)
    x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(u, x_dir)  # 确保y轴与x、u正交

    # 构建旋转矩阵（列向量为新坐标系的基向量）
    R = np.column_stack((x_dir, y_dir, u))

    # 步骤3：应用变换
    transformed_cloud = point_cloud @ R
    return transformed_cloud

def construct_grid(point_cloud, grid_width_count):
    """
    根据点云包围盒构造XOY平面的二维栅格，并记录每个栅格内的点云索引

    参数:
    point_cloud (np.ndarray): 点云数据，形状为(n, 3)
    grid_width_count (int): X方向的栅格数量，Y方向栅格数量会按比例计算

    返回:
    list[list[list]]: 二维列表表示的栅格，每个元素是落入该栅格的点云索引列表
    """
    if point_cloud.size == 0:
        return []

    # 计算点云在XOY平面的包围盒
    min_coords = np.min(point_cloud[:, :2], axis=0)  # X和Y的最小值
    max_coords = np.max(point_cloud[:, :2], axis=0)  # X和Y的最大值
    extent = max_coords - min_coords  # 包围盒范围

    # 处理退化情况（点云在一条线或一个点上）
    if np.allclose(extent, 0):
        grid = [[[]]]
        grid[0][0] = list(range(len(point_cloud)))
        return grid

    # 计算栅格数量和大小
    x_grid_count = grid_width_count
    y_grid_count = max(1, int(np.ceil(x_grid_count * extent[1] / extent[0])))
    cell_size = extent / np.array([x_grid_count, y_grid_count])

    # 初始化栅格结构
    grid = [[[] for _ in range(y_grid_count)] for _ in range(x_grid_count)]

    # 将每个点分配到对应的栅格
    for i, point in enumerate(point_cloud):
        x, y = point[:2]
        # 计算点在栅格中的索引（防止超出边界）
        grid_x = min(int((x - min_coords[0]) / cell_size[0]), x_grid_count - 1)
        grid_y = min(int((y - min_coords[1]) / cell_size[1]), y_grid_count - 1)
        grid[grid_x][grid_y].append(i)

    return grid


def grid_to_image(grid, point_cloud, colors, background_color=(0, 0, 0), use_height_max=True):
    """
    将栅格数据转换为图片

    参数:
    grid (list[list[list]]): 二维栅格，每个元素是点云索引列表
    point_cloud (np.ndarray): 点云数据，形状为(n, 3)
    colors (np.ndarray): 点云颜色，形状为(n, 3)或(n, 4)，取值范围0-255
    background_color (tuple): 背景颜色，RGB元组，默认黑色
    use_height_max (bool): 是否使用高程最大的点的颜色，否则使用平均颜色

    返回:
    PIL.Image: 生成的图片
    """
    x_size = len(grid)
    y_size = len(grid[0]) if x_size > 0 else 0

    # 创建背景图片
    image_data = np.zeros((y_size, x_size, len(background_color)), dtype=np.uint8)
    image_data[:] = background_color

    # 处理每个栅格
    for x in range(x_size):
        for y in range(y_size):
            indices = grid[x][y]
            if not indices:
                continue

            if use_height_max:
                # 选择高程(z坐标)最大的点
                heights = [point_cloud[i, 2] for i in indices]
                max_idx = indices[np.argmax(heights)]
                color = colors[max_idx]
            else:
                # 计算所有点的平均颜色
                color = np.mean(colors[indices], axis=0)

            # 确保颜色值在0-255范围内
            color = np.clip(color, 0, 255).astype(np.uint8)
            image_data[y, x] = color  # 注意PIL中y轴向下，与常规坐标系相反

    # # 创建PIL图片
    # if image_data.shape[2] == 3:
    #     image = Image.fromarray(image_data, 'RGB')
    # else:  # 4通道
    #     image = Image.fromarray(image_data, 'RGBA')

    return image_data


import numpy as np
from scipy import ndimage


def fill_background_with_interpolation(grid, backcolor, tolerance=5, sigma=1.0):
    """
    使用插值法填充栅格中的背景区域，保留非背景区域的原始颜色

    参数:
    grid (np.ndarray): 栅格数据，形状为(m, n, 3)，RGB颜色值
    backcolor (tuple): 背景色，RGB元组
    tolerance (int): 颜色容差，用于判断是否为背景色
    sigma (float): 高斯滤波的标准差，控制插值的平滑程度

    返回:
    np.ndarray: 填充后的栅格
    """
    # 确保输入是numpy数组
    grid = np.array(grid, dtype=np.float64)
    backcolor = np.array(backcolor, dtype=np.float64)

    # 创建背景掩码（True表示背景）
    mask = np.sqrt(np.sum((grid - backcolor) ** 2, axis=2)) <= tolerance

    # 对每个颜色通道分别进行插值
    filled_grid = grid.copy()
    for channel in range(3):
        # 创建临时数组，背景区域为0，非背景区域保持原值
        temp = grid[:, :, channel].copy()
        temp[mask] = 0

        # 使用高斯滤波进行插值
        blurred = ndimage.gaussian_filter(temp, sigma=sigma)

        # 创建权重掩码，背景区域权重为0，非背景区域权重为1
        weight = np.ones_like(temp)
        weight[mask] = 0
        blurred_weight = ndimage.gaussian_filter(weight, sigma=sigma)

        # 避免除以零
        valid = blurred_weight > 1e-8
        filled_channel = np.zeros_like(temp)
        filled_channel[valid] = blurred[valid] / blurred_weight[valid]

        # 仅替换背景区域的值
        filled_grid[:, :, channel][mask] = filled_channel[mask]

    # 确保颜色值在0-255范围内
    filled_grid = np.clip(filled_grid, 0, 255).astype(np.uint8)

    return filled_grid

def ply_to_image(ply_path, direct, grid_num, output_path, background_color=(0, 0, 0)):
    """
    将PLY格式点云转换为图像

    参数:
    ply_path (str): PLY文件路径
    direct (tuple): 投影方向向量(x, y, z)
    grid_num (int): 栅格数量
    output_path (str): 输出图像路径
    background_color (tuple): 背景颜色，默认为黑色(0, 0, 0)
    """
    # 1. 读取PLY文件并投影点云
    points, colors = read_ply(ply_path)
    # 1 投影转换。
    # projected_points = transform_point_cloud(points, (0,0,-1))

    # 2. 计算包围盒并栅格化
    grid_width_count = 1000
    grid = construct_grid(points, grid_width_count)

    # 3. 栅格转图像
    img = grid_to_image(grid, points, colors, (0, 0, 0), True)

    # 3.5 插值填充
    backcolor = (0, 0, 0)
    img_insert = fill_background_with_interpolation(img, backcolor, tolerance=5)

    # 4. 写出图片
    cv2.imencode('.jpg', img_insert)[1].tofile(output_path)
    print(f"图像已保存至: {output_path}")

def read_ply(filename):
    """读取PLY文件，返回点云坐标和颜色"""
    points = []
    colors = []

    # 以二进制模式打开文件
    with open(filename, 'rb') as f:
        # 读取头部行直到"end_header"
        header_lines = []
        line = f.readline()
        while line and b'end_header' not in line:
            header_lines.append(line.decode('utf-8', errors='ignore').strip())
            line = f.readline()

        # 解析头部信息
        vertex_count = 0
        property_types = []
        color_offset = None
        has_color = False

        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                # 记录属性类型和名称
                property_types.append((parts[1], parts[2]))
                if not has_color and parts[2] in ['red', 'diffuse_red']:
                    has_color = True
                    color_offset = len(property_types) - 1

        # 确定坐标和颜色的数据类型
        x_type = property_types[0][0] if len(property_types) > 0 else 'float'
        y_type = property_types[1][0] if len(property_types) > 1 else 'float'
        z_type = property_types[2][0] if len(property_types) > 2 else 'float'

        # 映射数据类型到struct格式字符
        type_map = {
            'char': 'b', 'uchar': 'B',
            'short': 'h', 'ushort': 'H',
            'int': 'i', 'uint': 'I',
            'float': 'f', 'double': 'd'
        }

        # 构建格式字符串
        format_str = '<'
        for p_type, _ in property_types:
            if p_type in type_map:
                format_str += type_map[p_type]
            else:
                # 默认使用float
                format_str += 'f'

        # 计算每条记录的大小
        record_size = struct.calcsize(format_str)

        # 读取顶点数据
        for _ in range(vertex_count):
            data = f.read(record_size)
            if len(data) < record_size:
                break  # 文件可能损坏

            # 解析数据
            values = struct.unpack(format_str, data)

            # 提取坐标
            x = values[0]
            y = values[1]
            z = values[2]
            points.append([x, y, z])

            # 提取颜色
            if has_color:
                # 检查颜色是否为浮点型
                is_float_color = (
                        property_types[color_offset][0] in ['float', 'double'] or
                        property_types[color_offset + 1][0] in ['float', 'double'] or
                        property_types[color_offset + 2][0] in ['float', 'double']
                )

                if is_float_color:
                    # 浮点型颜色(0.0-1.0)转换为uint8(0-255)
                    r = int(min(max(values[color_offset] * 255, 0), 255))
                    g = int(min(max(values[color_offset + 1] * 255, 0), 255))
                    b = int(min(max(values[color_offset + 2] * 255, 0), 255))
                else:
                    # 整型颜色
                    r = values[color_offset]
                    g = values[color_offset + 1]
                    b = values[color_offset + 2]

                colors.append([b, g, r])  # OpenCV使用BGR顺序
            else:
                # 如果没有颜色信息，默认为白色
                colors.append([255, 255, 255])

    # 使用float64提高精度，避免极小值问题
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8)

# 使用示例
if __name__ == "__main__":
    # 参数设置
    ply_file =r"H:\TestData\墙面计算\test_g-ai\48\30_colored_point_cloud.ply"  # 替换为实际的PLY文件路径
    projection_direction = (0, 0, 1)  # 沿Z轴方向投影
    grid_count = 200  # 栅格数量
    output_image = r"H:\TestData\墙面计算\test_g-ai\48\projection_result.png"
    background = (0, 0, 0)  # 黑色背景

    # 执行转换
    ply_to_image(ply_file, projection_direction, grid_count, output_image, background)
    # file_path = r"H:\TestData\墙面计算\test_g-ai\48\30_colored_point_cloud.ply",
    # r"H:\TestData\墙面计算\test_g-ai\48\projection_result.png"



