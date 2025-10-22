'''
    三维数据解析: 各种格式 点云, 网格
    1   读入
    2   解析
    3   写出
    -----------------
    薛凯 2025.04.15 更新
    注意:
    1\ 验证可靠的函数写好备注.
    2\
'''
import os
import plyfile
import numpy as np
from pypcd import pypcd  # 修改会报错

# 可能表示标签的字段
label_fields = ['class', 'scalar_class', 'label', 'scalar_label', 'labels', 'scalar_labels']


def np2pcd(points,labels):
    '''
    写出带标签的点
        :param points:输入点坐标ndarray
        :param labels:输入标签ndarray
        :param
        :return:pcd 返回pcd格式标签点云
        '''
    num_points = labels.size
    # 1 点云
    if labels.ndim < 2:
        labels = labels[:, np.newaxis]
    cloud = np.hstack((points, labels))
    cloud = cloud.astype(np.float32)
    # 2 写出
    metadata = {
        'version': '0.7',
        'fields': ['x', 'y', 'z', 'label'],
        'size': [4, 4, 4, 4],  # 数据类型大小 (float32)
        'type': ['F', 'F', 'F', 'F'],  # 数据类型 (float)
        'count': [1, 1, 1, 1],  # 每个字段的计数
        'width': num_points,
        'height': 1,
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],
        'points': num_points,
        'data': 'binary'  # 'ascii'  # 数据存储格式 (ascii 或 binary_compressed)
    }
    pcd = pypcd.PointCloud(metadata, cloud)
    return pcd


# ply解析
def parse_ply_file(file_path):
    """
    TODO: 新增参数，是否写出所有ply中参数，还是只写出标签。满足更广泛的应用
    ply文件解析,返回坐标\颜色\标签(ndarray)
    return coords, colors, labels   # ndarray, 无则对应返回None
    """
    coords, colors, labels = None,None,None
    with open(file_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertex_data = plydata['vertex'].data
        vertex_properties = vertex_data.dtype.names
        # 1）坐标
        if 'x' in vertex_properties and 'y' in vertex_properties and 'z' in vertex_properties:
            coords = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
            # 数据格式纠正
            if not np.issubdtype(coords.dtype, np.floating):
                coords = np.float32(coords)
        else:
            ValueError(f"【错误】{file_path}找不到坐标xyz")
        # 2）颜色
        if 'red' in vertex_properties and 'green' in vertex_properties and 'blue' in vertex_properties:
            colors = np.array([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
            if not np.issubdtype(colors.dtype, np.integer):
                colors = np.round(colors).astype(np.int32)
        else:
            print(f"{file_path}找不到颜色rgb")
        # 3）标签
        label_class = list(set(label_fields)&set(vertex_properties))  # 在label_fields中找同名标签名
        if len(label_class) == 0:
            print("【警告】读入的ply文件中不包含标签信息。")
        else:
            if len(label_class) >1:
                print(f"【警告】存在多个标签{label_class}，默认使用第一个标签。")
            labels = np.array(vertex_data[label_class[0]]).T
            labels = np.round(labels).astype(np.int32)

        # for label_item in label_fields:
        #     if label_item in vertex_properties:
        #         labels = np.array(vertex_data[label_item]).T
        #         labels = np.round(labels).astype(np.int32)
        #         break
        #     else:
        #         print("文件中不包含标签信息。")
    else:
        raise ValueError(f"点云{file_path}找不到数据")
    return coords, colors, labels

def parse_pcd_file(file_path):
    '''
    解析文件,返回coords, colors, labels信息.无信息则返回None
    pypcd 库
    注意:
    1\ 支持的标签名在全局变量label_fields中
    '''
    coords, colors, labels = None, None, None
    try:

        pc = pypcd.PointCloud.from_path(file_path)

        # 提取坐标信息
        if 'x' in pc.fields and 'y' in pc.fields and 'z' in pc.fields:
            x = pc.pc_data['x']
            y = pc.pc_data['y']
            z = pc.pc_data['z']
            coords = np.column_stack((x, y, z))
        else:
            print("PCD 文件中不包含坐标信息。")

        # 提取颜色信息
        if 'rgb' in pc.fields:
            colors = pc.pc_data['rgb']
        else:
            print("PCD 文件中不包含颜色信息。")

        # 提取标签信息
        # TODO：修改后未验证！！！
        label_class = list(set(label_fields) & set(pc.fields))  # 在label_fields中找同名标签名
        if len(label_class) == 0:
            print("【警告】读入的pcd文件中不包含标签信息。")
        else:
            if len(label_class) >1:
                print(f"【警告】存在多个标签{label_class}，默认使用第一个标签。")
            labels = pc.pc_data[label_class[0]]

        # for label_item in label_fields:
        #     if label_item in pc.fields:
        #         labels = pc.pc_data[label_item]
        #         break
        #     else:
        #         print("PCD 文件中不包含标签信息。")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return coords, colors, labels

def parse_3d_cloud_file(file_path):
    '''
    统一的解析接口,目前支持点云格式:['pcd', 'ply']
    '''
    _,file_extention = os.path.splitext(file_path)
    if file_extention == '.ply':
        return parse_ply_file(file_path)
    elif file_extention == '.pcd':
        return parse_pcd_file(file_path)
    else:
        raise TypeError(f'输入文件格式 {file_extention} ,未处理')

def write_ply_file(file_path,cloud_ndarray):
    '''
    读入的ndarray写出ply文件
    file_path 地址
    cloud_ndarray nd格式点云信息
    注意：
        1）当前写死格式【坐标、颜色、标签】，后面需要改为动态处理
        2) 二进制还是 ascii
    '''
    try:
        num_points = cloud_ndarray.shape[0]
        np.savetxt(file_path, cloud_ndarray, fmt='%f %f %f %d %d %d %i')
        # 定义 PLY 文件头
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property int red
property int green
property int blue
property int class
end_header
"""
        with open(file_path, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(header)
            f.write(old)
        print(f"合并后的带标签点云已成功保存到 {file_path}")

    except Exception as e:
        print(f"保存文件时出现错误: {e}")


import numpy as np
import struct


# 弃用：for处理太慢，使用write_ply_file_binary_batch批量处理替换
def write_ply_file_binary(file_path, cloud_ndarray):
    '''
    将带有坐标、颜色和标签的点云数据写入二进制PLY文件
    参数:
        file_path: 输出文件路径
        cloud_ndarray: numpy数组，格式为[x, y, z, r, g, b, class]
    注意:
        1) 读入数据目前写死！[x, y, z, r, g, b, class]
        2) 类别目前unchar存储（0-255），更多类别需要修改存储类型
    '''
    try:
        num_points = cloud_ndarray.shape[0]

        # 定义PLY文件头
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar class
end_header
"""
        # 写入文件头
        with open(file_path, 'wb') as f:
            f.write(header.encode('ascii'))

            # 写入二进制数据
            for i in range(num_points):
                data = struct.pack(
                    'fffBBBB',  # 格式说明符
                    np.float32(cloud_ndarray[i, 0]),  # x (float32)
                    np.float32(cloud_ndarray[i, 1]),  # y (float32)
                    np.float32(cloud_ndarray[i, 2]),  # z (float32)
                    np.uint8(cloud_ndarray[i, 3]),  # r (uint8)
                    np.uint8(cloud_ndarray[i, 4]),  # g (uint8)
                    np.uint8(cloud_ndarray[i, 5]),  # b (uint8)
                    np.uint8(cloud_ndarray[i, 6])  # class (uint8)
                )
                f.write(data)

        print(f"二进制PLY文件已成功保存到 {file_path}")

    except Exception as e:
        print(f"保存文件时出现错误: {e}")
        raise

def write_ply_file_binary_batch(file_path, cloud_ndarray,
                                chunk_size: int = 1000000):
    '''
    将带有坐标、颜色和标签的点云数据写入二进制PLY文件
    参数:
        file_path: 输出文件路径
        cloud_ndarray: numpy数组，格式为[x, y, z, r, g, b, class]
        chunk_size: 分块写入的点数（根据内存调整）
    注意:
        1) 读入数据目前写死！[x, y, z, r, g, b, class]
        2) 类别目前unchar存储（0-255），更多类别需要修改存储类型
    '''
    try:
        num_points = cloud_ndarray.shape[0]

        # 1 定义PLY文件头
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar class
end_header
"""
        # 2 定义数据：转换为结构化数组
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('cls', 'u1')
        ])
        structured_data = np.empty(num_points, dtype=dtype)
        structured_data['x'] = cloud_ndarray[:, 0].astype('f4')
        structured_data['y'] = cloud_ndarray[:, 1].astype('f4')
        structured_data['z'] = cloud_ndarray[:, 2].astype('f4')
        structured_data['r'] = cloud_ndarray[:, 3].astype('u1')
        structured_data['g'] = cloud_ndarray[:, 4].astype('u1')
        structured_data['b'] = cloud_ndarray[:, 5].astype('u1')
        structured_data['cls'] = cloud_ndarray[:, 6].astype('u1')

        # 3 分块写入（避免内存爆炸）
        with open(file_path, 'wb') as f:
            # 1) 写入文件头
            f.write(header.encode('ascii'))
            # 2) 分块处理
            for i in range(0, num_points, chunk_size):
                chunk = structured_data[i:i + chunk_size]
                f.write(chunk.tobytes())  # 直接写入二进制块
        print(f"已写入 {num_points:,} 点 | 文件大小: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"保存文件时出现错误: {e}")
        raise