import os
import argparse
import glob
import json
import time

import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import open3d as o3d
from scipy.spatial import KDTree
import pickle
import warnings
import torch
import points.points_io as points_io
import points.points_eval as points_eval
import path.path_process as path_process
from points.points_transform import grid_sample_idx
import label_dict

def val_pth(folder_path):
    '''
    验证生成的pth文件是否合格
    注意：
        1）兼容输入的文件名或文件夹名
    para:
        folder_path     遍历地址，写出：
    '''
    # folder_path = r'/home/xuek/桌面/Pointcept/data/bimtwins6Class20250730'
    ext = os.path.splitext(folder_path)[1]
    if ext == '.pth':
        filelist = [folder_path]
    else:
        filelist = path_process.get_files_by_format(folder_path, formats=['.pth'])
    for file in filelist:
        pth_file = file
        print(f"处理文件: {os.path.split(file)[1]}")
        # pth_file = r'/home/xuek/桌面/Pointcept/data/bimtwins6Class20250730/val/40GMGS.pth'
        pth_info = torch.load(pth_file)
        label = pth_info['semantic_gt20']
        points_eval.label_rate(label)
        # unique_labels, counts = np.unique(label, return_counts=True)
        # num = label.shape[0]
        # print(f"点云数量: {num}")
        # for labeli, counti in zip(unique_labels, counts):
        #     print(f"    标签: {labeli} ,数量: {counti} ,占比: {counti / num:.2%}")

    return

def cloudinfo_to_ptv3dict(cloud_info, sample_grid=0.4):
    '''
    输入一个点云信息字典，返回PTV3处理的数据字典
    （并进行将采样、计算点法向）
    注意：
        * 参考：/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/preprocess.py
        * 当前不支持标签读入！！！！，因为o3d
    return
        save_dict, idx_proj
    '''
    # 0 构造返回字典
    save_dict = dict()
    # 1 点云grid sample
    if not 'coords' in cloud_info:
        raise ValueError(f'【错误】输入点云没有坐标信息！')
    idx_sample = grid_sample_idx(cloud_info['coords'], grid_size=sample_grid)
    # 2 采样后点云构造
    # 2.1 坐标
    coords_sub = cloud_info['coords'][idx_sample]
    save_dict['coord'] = coords_sub
    # 2.2 颜色
    if 'colors' in cloud_info:
        colors_sub = cloud_info['colors'][idx_sample]
        save_dict['color']=colors_sub
    # 2.3 法向
    if 'normals' in cloud_info:
        if cloud_info['normals'] != None:
            normals_sub = cloud_info['normals'][idx_sample]
        else:
            radius = 0.5  # 搜索半径
            max_nn = 30  # 邻域内用于估算法线的最大点数
            cloud_o3d_sample = o3d.geometry.PointCloud()
            cloud_o3d_sample.points = o3d.utility.Vector3dVector(colors_sub)
            cloud_o3d_sample.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
            normals_sub = np.asarray(cloud_o3d_sample.normals)
        save_dict['normal'] = normals_sub

    # 2.4 原始标签
    if 'labels' in cloud_info:
        labels_sub = cloud_info['labels'][idx_sample]
    else:  # 构造假标签
        print("没有找到标签！")
        labels_sub = np.zeros((len(coords_sub),), dtype=np.int8)  # 创建一个假的标签值
    save_dict["semantic_gt20"] = labels_sub
    # 写出
    if False:
        cloud_ndarray=np.hstack([coords_sub,colors_sub,labels_sub.reshape(-1, 1)])
        file_out =r'/home/xuek/桌面/TestData/临时测试区/input_34PTY1_out.ply'
        write_ply_file(file_out, cloud_ndarray)
    return save_dict


def get_ply_file_info(ply_file, info_needed=['coords', 'colors', 'labels', 'normals'], label_process = None):
    '''
    从PLY文件中提取指定的信息
    Args:
        ply_file (str): PLY文件路径
        info_needed (list): 需要提取的信息列表，可包含'coords', 'colors', 'labels', 'normals'
        label_process (str, dict or None): 标签处理方式
            - 如果是字符串：表示目标标签名称，如'label12_V1'
            - 如果是字典：表示外部输入标签映射关系，如 { 0: [0], 1: [1], 2: [3], 3: [2]}
            - 如果是None：直接使用原始标签
    Returns:
        dict: 包含所需信息的字典，根据info_needed参数返回相应键值对
        
    Raises:
        ValueError: 当必需的信息缺失时抛出异常
    '''
    # 点云字典
    cloud_info = dict()
    points_dict = points_io.parse_cloud_to_dict(ply_file)
    # 1 坐标
    if 'coords' in info_needed and 'coords' in points_dict:
        coords = points_dict['coords']
        cloud_info['coords']=coords
    else:
        raise ValueError(f'没有坐标信息！')
    # 2 颜色
    if 'colors' in info_needed and 'colors' in points_dict:
        colors = points_dict['colors']
        cloud_info['colors'] = colors
    else:
        raise ValueError(f'没有颜色信息！')
    # 3 法向
    if 'normals' in info_needed:
        if 'normals' in points_dict:
            normals = points_dict['normals']
        else:
            normals = None  # 此处不处理，采样后再计算。
        cloud_info['normals'] = normals

    # 4 标签
    if 'labels' in info_needed:
        # 4.1 指定标签名
        if isinstance(label_process, str):
            label_name = label_process  # 写出的标签名
            # 0) 获取标签名列表,如{'label12_v1’,’label05_V1‘}
            labels_list = label_dict.LabelRegistry.labels_correct_from_ply(points_dict.keys())
            # 1) 计算最优标签映射
            label_input_name, label_mapping = label_dict.LabelRegistry.label_map(labels_list, label_name)
            labels = points_dict[label_input_name]
            # 2) 进行标签映射
            print(f'原始标签统计：')
            points_eval.label_rate(labels)
            new_labels = points_eval.label_change(labels, label_mapping)
            print(f'新标签统计：')
            points_eval.label_rate(new_labels)
            cloud_info['labels'] = new_labels
        # 4.2 指定标签映射
        elif isinstance(label_process, dict):
            # 1) 获取唯一标签名：判定为包含label、或class的字段  #  TODO： 筛选唯一标签的方法待优化
            lable_names = label_dict.LabelRegistry.get_labels_property_name(points_dict.keys())
            if len(lable_names) == 0:
                raise ValueError(f'没有标签信息！')
            elif len(lable_names) > 1:
                raise ValueError(f'多个标签信息！')
            label_name = lable_names.pop()
            labels = points_dict[label_name]
            # 2) 映射字典
            label_mapping = label_process
            # 3) 进行标签映射
            print(f'原始标签统计：')
            points_eval.label_rate(labels)
            new_labels = points_eval.label_change(labels, label_mapping)
            print(f'新标签统计：')
            points_eval.label_rate(new_labels)
            cloud_info['labels'] = new_labels
        else:
            # 1) 获取唯一标签名：判定为包含label、或class的字段  #  TODO： 筛选唯一标签的方法待优化
            lable_names = label_dict.LabelRegistry.get_labels_property_name(points_dict.keys())
            if len(lable_names) == 0:
                raise ValueError(f'没有标签信息！')
            elif len(lable_names) > 1:
                raise ValueError(f'多个标签信息！')
            label_name = lable_names.pop()
            labels = points_dict[label_name]
            # 3) 进行标签映射
            print(f'不转换标签，标签统计结果：')
            points_eval.label_rate(labels)
            cloud_info['labels'] = labels

    return cloud_info


def preprocess_BIMTwins(
        input_file, output_file,
        info_needed=['coords', 'colors', 'labels', 'normals'],
        label_process='label05_V1',
        sample_grid=0.4, val_result_pth=True
):

    # 1 获取需要的点云信息
    cloud_info = get_ply_file_info(input_file, info_needed, label_process)

    # 2 将采样、点云化，处理为ptv3输入格式
    # （注意：当前不再构造kdtree上采样树）
    save_dict = cloudinfo_to_ptv3dict(cloud_info, sample_grid=sample_grid)
    if True:  # train模式需要
        scene_id = os.path.splitext(os.path.split(input_file)[1])[0]
        save_dict['scene_id'] = scene_id
    # 写出采样后的点云的 pth文件
    torch.save(save_dict, output_file)

    if val_result_pth:
        print(f"验证pth数据标签（降采样结果）：", end="")
        val_pth(output_file)
    return

def preprocess_BIMTwins_batch(
        dataset_root, output_root,
        info_needed=['coords', 'colors', 'labels', 'normals'],
        label_process='label05_V1',
        sample_grid=0.4, val_result_pth=True
):
    # 1 数据地址
    # 查找所有ply文件
    sceneList = sorted(glob.glob(dataset_root + "/train/*.ply"))
    sceneList += sorted(glob.glob(dataset_root + "/val/*.ply"))
    sceneList += sorted(glob.glob(dataset_root + "/test/*.ply"))

    # 2 创建输出文件地址
    train_output_dir = os.path.join(output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # 3 批量处理
    time_start0 = time.time()
    for input_file in sceneList:
        time_starti = time.time()
        subFoler = os.path.split(os.path.split(input_file)[0])[1]  # 'train'
        scene_id = os.path.splitext(os.path.split(input_file)[1])[0]  # 'JXY'
        output_file = os.path.join(output_root, subFoler, f"{scene_id}.pth")
        print(f"Processing: {scene_id} in {subFoler}")

        preprocess_BIMTwins(
            input_file, output_file,
            info_needed=info_needed,
            label_process=label_process,
            sample_grid=sample_grid, val_result_pth=val_result_pth
        )

        print(f"场景 {scene_id} 耗时：{time.time() - time_starti}")
        print(" ")
    print(f"预处理 总耗时：{time.time() - time_start0}")

def preprocess_BIMTwins_备份1130(
        input_file, output_file,
        generate_normals=True, sample_grid=0.4,
        label_dict={},
        val_result_pth=True,
        create_fake_label=False,
):
    '''
    brief:
        预处理：读入原始1个点云数据，生成ptv3训练输入的降采样点云.pth
        注意：
            1）输入是点云地址，写出pth文件地址
            2）label_dict要求如下描述。
    paras:
        input_file          输入原始点云
        output_file         写出pth文件
        generate_normals    是否写出法向
        sample_grid         采样密度
        label_dict          新旧字典映射：{新标签1：[旧标签1，就标签2, ...], },ex: label_dict={0:[0,1],1:[2,3]}；
                                1）未修改标签则置为0，会提示。
                                2）空，表示不修改标签，会提示。
        val_result_pth      是否验证生成的pth文件
        create_fake_label   如果没有标签，则创造假标签
    '''
    # 1 获取点云信息：坐标、颜色、标签
    try:
        coords, colors, labels = points_io.parse_ply_file(input_file)
    except ValueError as e:
        raise ValueError(f'发生错误{e}')
    if not isinstance(coords, np.ndarray):
        raise ValueError(f'没有点')
    elif not isinstance(labels, np.ndarray):
        if not create_fake_label:
            raise ValueError(f'输入数据没有标签！（如需构造虚假标签，修改变量“create_fake_label”）')
        else:
            print(f"【警告】未找到标签信息，此处将构造虚假标签！")
            labels = np.zeros(len(coords), int)

    # 2 标签替换
    print(f'原始标签统计：')
    points_eval.label_rate(labels)
    new_labels = points_eval.label_change(labels, label_dict)
    print(f'新标签统计：')
    points_eval.label_rate(new_labels)

    # 4 构造cloud_info点云信息
    cloud_info = dict(coords=coords, colors=colors, labels=new_labels)

    # 5 将采样、点云化，处理为ptv3输入格式
    # （注意：当前不再构造kdtree上采样树）
    save_dict = cloudinfo_to_ptv3dict(cloud_info, generate_normals=generate_normals, sample_grid=sample_grid)
    if True:  # train模式需要
        scene_id = os.path.splitext(os.path.split(input_file)[1])[0]
        save_dict['scene_id'] = scene_id
    # 写出采样后的点云的 pth文件
    torch.save(save_dict, output_file)

    if val_result_pth:
        print(f"验证pth数据标签（降采样结果）：", end="")
        val_pth(output_file)
    return

def preprocess_BIMTwins_batch_备份1130(
        dataset_root, output_root,
        generate_normals=True, sample_grid=0.4,
        label_dict={},
        val_result_pth=True,
        create_fake_label=False,
):
    # 1 数据地址
    # 查找所有ply文件
    sceneList = sorted(glob.glob(dataset_root + "/train/*.ply"))
    sceneList += sorted(glob.glob(dataset_root + "/val/*.ply"))
    sceneList += sorted(glob.glob(dataset_root + "/test/*.ply"))

    # 2 创建输出文件地址
    train_output_dir = os.path.join(output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # 3 批量处理
    time_start0 = time.time()
    for input_file in sceneList:
        time_starti = time.time()
        subFoler = os.path.split(os.path.split(input_file)[0])[1]  # 'train'
        scene_id = os.path.splitext(os.path.split(input_file)[1])[0]  # 'JXY'
        output_file = os.path.join(output_root, subFoler, f"{scene_id}.pth")
        print(f"Processing: {scene_id} in {subFoler}")
        preprocess_BIMTwins(
            input_file, output_file,
            generate_normals, sample_grid=sample_grid,
            label_dict=label_dict,
            val_result_pth=val_result_pth,
            create_fake_label=val_result_pth,
        )
        print(f"场景 {scene_id} 耗时：{time.time() - time_starti}")
        print(" ")
    print(f"预处理 总耗时：{time.time() - time_start0}")

'''
参数说明：
    dataset_root    输入数据（输入时已经按照train\ val\ test\ 分配好）
    output_root     输出数据地址
    parse_normals   是否使用点云法向
'''
if __name__ == "__main__":

    if True:  # 批量处理
        dataset_root = r'/media/xuek/Data210/数据集/临时测试区/训练集_temp'
        output_root = r'/media/xuek/Data210/数据集/临时测试区/训练集_temp_out'
        info_needed = ['coords', 'colors', 'labels', 'normals']
        '''
        label_process 调用说明：
            方法1： label_process = 'label05_V1'  
                直接说明调用的标签版本，前提是数据集均为统一命名方式。
                找不到则会选择最佳转换方式。
            方法2： label_process = label_dict.LabelRegistry.label_mappings['label12_V1_to_label05_V1'] 
                直接给出映射字典，会调用唯一标签执行映射。
                存在多标签会报错
            方法3： label_process = None 不修改标签，直接使用默认标签执行。
        '''
        # label_process = 'label05_V1'
        label_process = label_dict.LabelRegistry.label_mappings['label12_V1_to_label05_V1']
        # label_process = None
        sample_grid = 0.4  # 采样大小：注意此处写死
        preprocess_BIMTwins_batch(
            dataset_root, output_root,
            info_needed=info_needed,
            label_process=label_process,
            sample_grid=sample_grid, val_result_pth=True
        )


    # 预处理数据  # 弃用
    if False:  # 单个数据
        time_starti = time.time()
        input_file =r'/media/xuek/Data210/数据集/训练集/重建数据_版本2025.10.15/train/06XCNC.ply'
        output_file = r'/home/xuek/桌面/TestData/临时测试区/重建数据_版本2025.10.15_weight20251113/test.pth'
        sample_grid = 0.4  # 采样大小：注意此处写死
        label_dict_input = label_dict.label_map_l12v1_to_l05v1
        preprocess_BIMTwins(
            input_file, output_file,
            generate_normals=True, sample_grid=sample_grid,
            label_dict=label_dict_input,
            val_result_pth=True,
            create_fake_label=False,
        )
        print(f"场景  耗时：{time.time() - time_starti}")
    if False:  # 批量处理（老版本）  # 弃用
        # 0 参数修改
        dataset_root = r'/media/xuek/备份盘/TempProcess/20251125数据集/LABEL12'
        output_root = r'/home/xuek/桌面/TestData/PTV3_data/重建数据_版本2025.10.15_5class_1125'
        sample_grid = 0.4  # 采样大小：注意此处写死
        label_dict_input = label_dict.label_map_l04v1_to_l05v1
        preprocess_BIMTwins_batch(
                dataset_root, output_root,
                generate_normals=True, sample_grid=0.4,
                label_dict=label_dict_input,
                val_result_pth=True,
                create_fake_label=False,
        )

