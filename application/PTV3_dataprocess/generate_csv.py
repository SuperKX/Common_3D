'''
生成点云统计CSV文件
输入点云数据，输出CSV文件包含：
1. 所在分组（train/val/test）
2. 场景名
3. 点云数量（总点云、各类别数量）
4. 数据类别占比（各类别点云比例）
5. 分组类别占比（该数据中类别i点云数量/该分组中所有类别i的数量）
'''
import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import copy

# 导入必要的模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import points.points_eval as points_eval
import path.path_process as path_process
from application.PTV3_dataprocess import data_dict

def generate_point_cloud_statistics_csv(data_folder, output_csv):
    '''
    生成点云统计CSV文件

    参数:
        data_folder: 数据文件夹路径（包含train/val/test子文件夹）
        output_csv: 输出CSV文件路径

    返回:
        DataFrame包含所有统计信息
    '''
    # 类别名称映射
    label_names = {
        0: "背景类",
        1: "建筑类",
        2: "车辆类",
        3: "高植被类",
        4: "低植被类"
    }

    # 子文件夹
    sub_folders = ['train', 'val', 'test']

    # 存储所有场景的数据
    all_scenes_data = []

    # 存储每个分组的类别总数，用于计算分组类别占比
    group_category_totals = defaultdict(lambda: defaultdict(int))

    # 遍历每个分组
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(data_folder, sub_folder)
        if not os.path.exists(sub_folder_path):
            continue

        # 获取所有.pth文件
        filelist = path_process.get_files_by_format(sub_folder_path, formats=['.pth'], return_full_path=False)

        # 遍历文件
        for file in filelist:
            # 获取场景名称
            file_name_raw = os.path.splitext(file)[0]
            if file_name_raw.endswith('_1216'):
                scene_name = file_name_raw[:-5]
            else:
                scene_name = file_name_raw

            file_path = os.path.join(sub_folder_path, file)

            # 加载点云数据
            import torch
            pth_info = torch.load(file_path)
            labels = pth_info['semantic_gt20']

            # 计算标签统计
            scene_data = {
                '分组': sub_folder,
                '场景名': scene_name,
                '总点云': labels.size
            }

            # 计算各类别数量和占比
            label_info = points_eval.label_rate(labels)

            # 各类别数量和占比
            category_counts = {}
            category_ratios = {}

            for label_id, info in label_info.items():
                category_name = label_names.get(label_id, f"类别{label_id}")
                category_counts[category_name] = info['数量']
                category_ratios[category_name] = info['比例']

                # 累计到分组类别总数
                group_category_totals[sub_folder][category_name] += info['数量']

            # 添加各类别数量到场景数据
            for category_name in label_names.values():
                scene_data[f'数量_{category_name}'] = category_counts.get(category_name, 0)
                scene_data[f'占比_{category_name}'] = category_ratios.get(category_name, 0.0)

            all_scenes_data.append(scene_data)

    # 计算分组类别占比并添加到数据中
    for scene_data in all_scenes_data:
        group = scene_data['分组']
        for category_name in label_names.values():
            scene_count = scene_data[f'数量_{category_name}']
            group_total = group_category_totals[group][category_name]
            scene_data[f'分组占比_{category_name}'] = scene_count / group_total if group_total > 0 else 0.0

    # 转换为DataFrame
    df = pd.DataFrame(all_scenes_data)

    # 重新排列列的顺序
    columns_order = ['分组', '场景名', '总点云']

    # 添加数量列
    for category_name in label_names.values():
        columns_order.append(f'数量_{category_name}')

    # 添加占比列
    for category_name in label_names.values():
        columns_order.append(f'占比_{category_name}')

    # 添加分组占比列
    for category_name in label_names.values():
        columns_order.append(f'分组占比_{category_name}')

    # 确保所有列都存在
    for col in columns_order:
        if col not in df.columns:
            df[col] = 0

    # 重新排列列
    df = df[columns_order]

    # 保存到CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"CSV文件已保存到: {output_csv}")
    print(f"共处理 {len(df)} 个场景")

    return df


def generate_point_cloud_statistics_csv_with_ply(ply_folder, output_csv, label_field='class_class'):
    '''
    从PLY文件夹生成点云统计CSV文件

    参数:
        ply_folder: PLY文件所在文件夹路径
        output_csv: 输出CSV文件路径
        label_field: PLY文件中的标签字段名

    返回:
        DataFrame包含所有统计信息
    '''
    import points.points_io as points_io

    # 类别名称映射
    label_names = {
        0: "背景类",
        1: "建筑类",
        2: "车辆类",
        3: "高植被类",
        4: "低植被类"
    }

    # 存储所有场景的数据
    all_scenes_data = []

    # 存储每个分组的类别总数（这里只有一个文件夹，当作一个分组）
    group_category_totals = defaultdict(int)

    # 获取所有PLY文件
    filelist = path_process.get_files_by_format(ply_folder, formats=['.ply'], return_full_path=False)

    # 遍历文件
    for file in filelist:
        # 获取场景名称
        scene_name = os.path.splitext(file)[0]

        file_path = os.path.join(ply_folder, file)

        try:
            # 读取PLY文件
            points_dict = points_io.parse_cloud_to_dict(file_path)

            # 获取标签
            if label_field not in points_dict:
                print(f"警告：文件 {file} 中没有找到标签字段 {label_field}")
                continue

            labels = points_dict[label_field]
            labels = np.array(labels)

            # 计算标签统计
            scene_data = {
                '分组': 'single_group',  # 单个文件夹作为一个分组
                '场景名': scene_name,
                '总点云': labels.size
            }

            # 计算各类别数量和占比
            label_info = points_eval.label_rate(labels)

            # 各类别数量和占比
            category_counts = {}
            category_ratios = {}

            for label_id, info in label_info.items():
                category_name = label_names.get(label_id, f"类别{label_id}")
                category_counts[category_name] = info['数量']
                category_ratios[category_name] = info['比例']

                # 累计到分组类别总数
                group_category_totals[category_name] += info['数量']

            # 添加各类别数量到场景数据
            for category_name in label_names.values():
                scene_data[f'数量_{category_name}'] = category_counts.get(category_name, 0)
                scene_data[f'占比_{category_name}'] = category_ratios.get(category_name, 0.0)

            all_scenes_data.append(scene_data)

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    # 计算分组类别占比并添加到数据中
    for scene_data in all_scenes_data:
        for category_name in label_names.values():
            scene_count = scene_data[f'数量_{category_name}']
            group_total = group_category_totals[category_name]
            scene_data[f'分组占比_{category_name}'] = scene_count / group_total if group_total > 0 else 0.0

    # 转换为DataFrame
    df = pd.DataFrame(all_scenes_data)

    # 重新排列列的顺序
    columns_order = ['分组', '场景名', '总点云']

    # 添加数量列
    for category_name in label_names.values():
        columns_order.append(f'数量_{category_name}')

    # 添加占比列
    for category_name in label_names.values():
        columns_order.append(f'占比_{category_name}')

    # 添加分组占比列
    for category_name in label_names.values():
        columns_order.append(f'分组占比_{category_name}')

    # 确保所有列都存在
    for col in columns_order:
        if col not in df.columns:
            df[col] = 0

    # 重新排列列
    df = df[columns_order]

    # 保存到CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"CSV文件已保存到: {output_csv}")
    print(f"共处理 {len(df)} 个场景")

    return df


if __name__ == '__main__':
    # 示例用法1：处理PTH文件
    # data_folder = r'/media/xuek/Data210/数据集/训练集/重建数据_动态维护_pth'
    # output_csv = r'/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/点云统计信息.csv'
    # df = generate_point_cloud_statistics_csv(data_folder, output_csv)

    # 示例用法2：处理PLY文件
    ply_folder = r'/media/xuek/Data210/数据集/训练集/重建数据_动态维护_ply'
    output_csv = r'/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/点云统计信息_ply.csv'
    df = generate_point_cloud_statistics_csv_with_ply(ply_folder, output_csv, label_field='label_label05_V1')

    # 打印前几行数据
    print("\n前5行数据:")
    print(df.head())

    # 打印统计信息
    print("\n统计信息:")
    print(df.describe())