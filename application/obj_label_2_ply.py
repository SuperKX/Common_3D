import open3d as o3d
import numpy as np

import mesh.mesh_common as mesh_fuc


# txt 解析标签：当前不用了
def load_labels(wrp_file):
    """
    读取 wrp标签 文件，解析出每个类别的面片编号。
    """
    class_labels = {}  # 字典存储
    current_class = None

    with open(wrp_file, 'r') as f:
        for line in f:
            line = line.strip()  # 移除行尾的空白字符
            if line.startswith("className"):
                current_class = line.split()[1]
                class_labels[current_class] = []  # 初始化类别列表
            elif line.isdigit():  # 数字即为编号
                class_labels[current_class].append(int(line))

    return class_labels


if __name__ == '__main__':

    # 1 标签解析
    wrp_file = r'E:\LabelScripts\testdata\wraptest\out\wraptest.txt'
    class_labels = load_labels(wrp_file)
    face_list = class_labels['wigwam']
    face_list = (np.array(face_list) - 1).tolist()  # 编号从1开始

    # 2 obj-seg
    input_obj_path = r'E:\LabelScripts\testdata\wraptest\out\wraptest\wraptest.obj'
    mesh = o3d.io.read_triangle_mesh(input_obj_path, True)


    mesh_seg = mesh_fuc.crop_mesh_with_face_ids(mesh, face_list)
    # outpath = r'H:\commonFunc_3D\testdata\seg16_2.obj'
    # # o3d.io.wrie_triangle_mesh(mesh,)