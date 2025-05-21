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
from pypcd import pypcd  # 修改回报错

# 可能表示标签的字段
label_fields = ['class', 'scalar_class', 'label', 'scalar_label', 'labels', 'scalar_labels']


# ply解析
def parse_ply_file(file_path):
    """
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
        for label_item in label_fields:
            if label_item in vertex_properties:
                labels = np.array(vertex_data[label_item]).T
                labels = np.round(labels).astype(np.int32)
                break
            else:
                print("文件中不包含标签信息。")
        # if 'class' in vertex_properties or 'scalar_class' in vertex_properties:
        #     if 'class' in vertex_properties:
        #         labels = np.array(vertex_data['class']).T
        #     else:
        #         labels = np.array(vertex_data['scalar_class']).T
        #     if not np.issubdtype(labels.dtype, np.integer):
        #         labels = np.round(labels).astype(np.int32)
        # else:
        #     print(f"【警告】{file_path}找不到标签值class、或scalar_class")
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
        for label_item in label_fields:
            if label_item in pc.fields:
                labels = pc.pc_data[label_item]
                break
            else:
                print("PCD 文件中不包含标签信息。")

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
