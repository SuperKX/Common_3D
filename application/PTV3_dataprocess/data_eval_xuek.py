import pandas as pd
import numpy as np
import os
# 导入data_dict和label_dict
from application.PTV3_dataprocess import data_dict, label_dict
import points.points_io as points_io
import points.points_eval as points_eval

def generate_scene_group_csv(output_csv=None, data_version=None, data_folder=None, label_version='label05_V1'):
    '''
    根据data_dict.py中的data_version20240905创建场景分组CSV表格

    参数:
        output_csv: 输出CSV文件路径（可选）
        data_version: 数据版本字典（可选，默认使用data_version20240905）
        data_folder: 数据文件夹路径（可选，用于统计点云数量）
        label_version: 标签版本（可选，默认使用label05_V1）

    返回:
        DataFrame包含分组和场景名信息，以及点云数量统计（如果提供data_folder）
    '''
    # 如果没有指定data_version，使用默认版本
    if data_version is None:
        data_version = data_dict.data_version20240905

    # 获取标签定义
    if label_version in label_dict.LabelRegistry.label_def:
        label_def = label_dict.LabelRegistry.label_def[label_version]
        # 根据标签定义生成类别列名
        category_columns = [label_def[i] for i in sorted(label_def.keys())]
    else:
        raise ValueError(f"未找到标签版本: {label_version}")

    print(f"\n生成场景分组CSV表格...")
    print(f"数据版本: data_version20240905")
    print(f"标签版本: {label_version}")
    if data_folder:
        print(f"数据文件夹: {data_folder}")

    # 存储所有数据
    all_data = []
    # points_num_class = {}
    # label_def

    # 1 遍历每个分组
    for group_name, scene_set in data_version.items():
        # 遍历该分组中的每个场景
        for scene_name in scene_set:
            # 1.1 场景基本信息
            row_data = {
                '分组': group_name,
                '场景名': scene_name
            }
            if data_folder:  # 如果提供了数据文件夹，统计点云数量
                ply_path = os.path.join(data_folder, f"{scene_name}.ply")

                if not os.path.exists(ply_path):
                    raise FileNotFoundError(f"未找到点云文件 {ply_path}")
                points_dict = points_io.parse_cloud_to_dict(ply_path)
                label_version_plystyle = 'label_' + label_version  # ply 文件前缀
                if not label_version_plystyle in points_dict:
                    raise ValueError(f"{scene_name}.ply 中没有{label_version}标签信息")
                labels = points_dict[label_version_plystyle]
                # points_num_info = points_eval.file_label_info(ply_path, label_version)

                # 1.2 点云数量
                row_data['总点云数量'] = len(labels)
                label_counts = {}
                for label_id, label_name in label_def.items():
                    count = np.sum(labels == label_id)
                    label_counts[label_name] = count
                for cat_name in category_columns:  # 类别数量
                    row_data[cat_name] = label_counts.get(cat_name, 0)
                # 1.3 点云内数据占比
                row_data['数据占比'] = 1
                for cat_name in category_columns:  # 类别比例
                    row_data['rt1_'+cat_name] = row_data[cat_name]/row_data['总点云数量']

                all_data.append(row_data)

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    # 2 按分组排序，然后按场景名排序
    if len(df) > 0:
        # 按分组顺序排序（train -> val -> test）
        group_order = {'train': 0, 'val': 1, 'test': 2}
        df['_group_order'] = df['分组'].map(group_order)
        df = df.sort_values(['_group_order', '场景名'])  # 优先级排序，分组>场景名
        df = df.drop('_group_order', axis=1)
        df = df.reset_index(drop=True)

    # 3 保存到CSV
    if output_csv and len(df) > 0:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n场景分组CSV已保存到: {output_csv}")

    # 打印统计信息
    if len(df) > 0:
        print("\n分组统计:")
        for group_name in data_version.keys():
            count = len(data_version[group_name])
            print(f"  {group_name}: {count} 个场景")
        print(f"  总计: {len(df)} 个场景")

    return df

if __name__ == '__main__':
    output_csv =r'H:\commonFunc_3D\application\PTV3_dataprocess/输出测试.csv'
    label_folder = r'J:\DATASET\BIMTwins\版本备份\多标签_动态维护版'
    generate_scene_group_csv(output_csv, data_folder=label_folder)