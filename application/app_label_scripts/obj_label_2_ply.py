'''
功能：批量obj打标签并点云化。
    ● obj及标签pth文件一一对应。
    ● 批量写出采样点云的ply文件，不写出中间文件。
xuek 2025.04.22
'''

import os
import open3d as o3d
import numpy as np
import pickle
import points.points_io as pts_io
import mesh.mesh_common as mesh_fuc



def read_dict_to_binary(filename):
    '''
    解析pth格式的字典，
    注意：
        1）wrp中标签下标从1开始，此处-1，跟面片下标对齐。
        2）默认每个标签的面片先排序。
    '''
    data = dict()
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print("success read")
    except Exception as e:
        print("wrong read!")
    # 排序，优化
    for key, val in data.items():  # wigwam
        val.sort()
        val = (np.array(val) - 1).tolist()  # 编号从1开始
        data[key] = val
    return data

# txt 解析标签：当前不用了
def load_labels(wrp_file):
    """
    读取 wrp标签 文件(txt)，解析出每个类别的面片编号。
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



def label_ply(class_labels, mesh):
    ''''
    输入mesh 和标签字典，写出打标签的点云
    class_labels 标签字典
    mesh 输入的网格
    '''
    global min_point_Num
    global subdense
    # 优化：并行处理！！
    merged_cloud = np.empty((0, 7))
    for key, face_list in class_labels.items():  # wigwam
        if len(face_list) == 0:
            continue
        if key not in classlist:
            print(f"【注意】：该标签 {key} 未统计！")
        # 4 面片裁剪
        mesh_seg = mesh_fuc.crop_mesh_with_face_ids(mesh, face_list)
        # 5 点云化
        pcd_seg = mesh_fuc.mesh_2_points(mesh_seg, subdense, min_point_Num)
        if len(pcd_seg.colors) <= 0:
            print(f"没有颜色，查看是否有贴图！")
        # 6 构造np格式点云
        coord_seg = np.hstack((np.array(pcd_seg.points), np.array(pcd_seg.colors) * 255))
        lablei = label_list[classlist.index(key)]
        points_seg_np = np.hstack((coord_seg, np.full((coord_seg.shape[0], 1), lablei)))  # x,y,z, r,g,b, label
        # 7 分块点云合并
        merged_cloud = np.vstack((merged_cloud, points_seg_np))  # shape(N,7)
    return merged_cloud



if __name__ == '__main__':

    #0  配置信息
    # 1）标签
    classlist = ["background", "building", "wigwam", "car", "vegetation", "farmland",
                 "shed", "stockpiles", "bridge", "pole", "others", "grass"]
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # 2）地址
    path_env = r"H:\commonFunc_3D\testdata\1_label"
    inputPath_obj = path_env + r"\obj"
    inputPath_pth = path_env + r"\pth"
    outputPath = path_env + r"\ply"
    # 3）采样参数
    global min_point_Num
    min_point_Num = 1  # 分块保证的最小采样点数量
    global subdense
    subdense = 0.1  # gridsample密度

    # 合法性判断
    if not os.path.exists(inputPath_obj) or not os.path.exists(inputPath_pth):
        raise FileNotFoundError("找不到输入路径！")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    # 逐个文件处理（优化：可并行）
    for pth_file in os.listdir(inputPath_pth):
        # 1 处理地址
        filename, ext = os.path.splitext(pth_file)
        if ext != '.pth':
            continue
        obj_file = os.path.join(inputPath_obj, filename, filename+'.obj')  # 读入的obj文件
        pth_fileP = os.path.join(inputPath_pth,pth_file)  # 读入的标签文件
        # 点云写出地址
        cloud_file = os.path.join(outputPath, filename+'.ply')
        # 2 解析 标签.pth
        class_labels = read_dict_to_binary(pth_fileP)
        # 3 解析 mesh
        mesh = o3d.io.read_triangle_mesh(obj_file, True)
        # 4 转点云
        merged_cloud = label_ply(class_labels, mesh)
        # 8 写出点云
        pts_io.write_ply_file(cloud_file, merged_cloud)
