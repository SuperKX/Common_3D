import pandas as pd
import numpy as np
import os
# 导入data_dict和label_dict
from application.PTV3_dataprocess import data_dict, label_dict
import points.points_io as points_io
import points.points_eval as points_eval

def generate_scene_group_csv(output_csv=None, data_version=None, data_folder=None, label_version='label05_V1',
                             pred_folder=None, label_version_pred='label05_V1'):
    '''
    根据data_dict.py中的data_version20240905创建场景分组CSV表格

    参数:
        output_csv: 输出CSV文件路径（可选）
        data_version: 数据版本字典（可选，默认使用data_version20240905）
        data_folder: 数据文件夹路径（可选，用于统计点云数量）
        label_version: 标签版本（可选，默认使用label05_V1）
        pred_folder: 推理输出数据文件夹路径（可选，用于计算评估指标）
        label_version_pred: 推理数据标签版本（可选，默认使用label05_V1）

    返回:
        DataFrame包含分组和场景名信息，以及点云数量统计（如果提供data_folder）
        和评估指标（如果提供pred_folder）
    '''
    # 如果没有指定data_version，使用默认版本
    if data_version is None:
        data_version = data_dict.data_version20240905

    # 获取标签定义
    if label_version in label_dict.LabelRegistry.label_def:
        label_def = label_dict.LabelRegistry.label_def[label_version]
        # 根据标签定义生成类别列名
        category_columns = [label_def[i] for i in sorted(label_def.keys())]
        class_num = len(label_def)
    else:
        raise ValueError(f"未找到标签版本: {label_version}")

    print(f"\n生成场景分组CSV表格...")
    print(f"数据版本: data_version20240905")
    print(f"标签版本: {label_version}")
    if data_folder:
        print(f"数据文件夹: {data_folder}")
    if pred_folder:
        print(f"推理文件夹: {pred_folder}")
        print(f"推理标签版本: {label_version_pred}")

    # 存储所有数据
    all_data = []

    # 各类别数量统计
    class_sum = {'train':{}, 'val':{}, 'test':{}}
    for cat_name in category_columns:
        class_sum['train'][cat_name] = 0
        class_sum['val'][cat_name] = 0
        class_sum['test'][cat_name] = 0
    class_sum['train']['分组点云总数'] = 0
    class_sum['val']['分组点云总数'] = 0
    class_sum['test']['分组点云总数'] = 0


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
                row_data['点云数量'] = len(labels)
                label_counts = {}
                for label_id, label_name in label_def.items():
                    count = np.sum(labels == label_id)
                    label_counts[label_name] = count
                for cat_name in category_columns:  # 类别数量
                    row_data[cat_name] = label_counts.get(cat_name, 0)
                    class_sum[group_name][cat_name] += row_data[cat_name]   # 各类别数量占比
                # 1.3 点云内数据占比
                row_data['数据占比'] = 1
                for cat_name in category_columns:  # 类别比例
                    row_data['drt_'+cat_name] = row_data[cat_name]/row_data['点云数量']

            all_data.append(row_data)
    # 1.4 计算分组内类别比例
    for cat_name in category_columns:
        class_sum['train']['分组点云总数'] += class_sum['train'][cat_name]
        class_sum['val']['分组点云总数'] += class_sum['val'][cat_name]
        class_sum['test']['分组点云总数'] += class_sum['test'][cat_name]
    for data_row in all_data:
        data_row['分组占比-总点云'] = data_row['点云数量']/class_sum[data_row['分组']]['分组点云总数']
        for cat_name in category_columns:
            data_row['grt_' + cat_name] = data_row[cat_name] / class_sum[data_row['分组']][cat_name]

    # 1.5 如果提供了推理文件夹，计算评估指标
    if pred_folder:
        for data_row in all_data:
            scene_name = data_row['场景名']
            label_path = os.path.join(data_folder, f"{scene_name}.ply")
            pred_path = os.path.join(pred_folder, f"{scene_name}.ply")

            if not os.path.exists(pred_path):
                print(f"警告: 未找到推理文件 {pred_path}，跳过评估")
                data_row['平均召回率'] = np.nan
                data_row['平均精确率'] = np.nan
                data_row['平均iou'] = np.nan
                # 各类别指标设为NaN
                for cat_name in category_columns:
                    data_row[f'{cat_name}_recall'] = np.nan
                    data_row[f'{cat_name}_precision'] = np.nan
                    data_row[f'{cat_name}_iou'] = np.nan
                # grass混淆度设为NaN
                data_row['grass2background'] = np.nan
                data_row['grass2vegetation'] = np.nan
                data_row['background2grass'] = np.nan
                data_row['vegetation2grass'] = np.nan
            else:
                # 读取标签数据
                labeled_dict = points_io.parse_cloud_to_dict(label_path)
                label_version_plystyle = 'label_' + label_version
                if not label_version_plystyle in labeled_dict:
                    raise ValueError(f"{scene_name}.ply 中没有{label_version}标签信息")
                labels = labeled_dict[label_version_plystyle]
                # 读取推理数据
                pred_dict = points_io.parse_cloud_to_dict(pred_path)
                label_version_pred_plystyle = 'label_' + label_version_pred
                if not label_version_pred_plystyle in pred_dict:
                    raise ValueError(f"{scene_name}.ply 中没有{label_version_pred}标签信息")
                labels_pred = pred_dict[label_version_pred_plystyle]

                # 计算混淆矩阵
                score_matrix = points_eval.matrix_eval(labels, labels_pred, class_num)

                # 计算评估指标
                score_rec, score_pre, score_iou = points_eval.eval_result(score_matrix, class_num)

                # 计算平均指标
                data_row['平均召回率'] = np.mean(score_rec)
                data_row['平均精确率'] = np.mean(score_pre)
                data_row['平均iou'] = np.mean(score_iou)

                # 各类别指标
                for i, cat_name in enumerate(category_columns):
                    data_row[f'{cat_name}_recall'] = score_rec[i]
                    data_row[f'{cat_name}_precision'] = score_pre[i]
                    data_row[f'{cat_name}_iou'] = score_iou[i]

                # 1.6 混淆度计算（grass与background、vegetation的混淆比例）
                # 获取grass、background、vegetation的标签ID
                grass_id = background_id = vegetation_id = None
                for label_id, label_name in label_def.items():
                    if label_name == 'grass':
                        grass_id = label_id
                    elif label_name == 'background':
                        background_id = label_id
                    elif label_name == 'vegetation':
                        vegetation_id = label_id

                if grass_id is not None and background_id is not None and vegetation_id is not None:
                    # grass被误分类为background的比例
                    grass_true_count = np.sum(score_matrix[grass_id, :])
                    if grass_true_count > 0:
                        data_row['grass2background'] = score_matrix[grass_id][background_id] / grass_true_count
                    else:
                        data_row['grass2background'] = np.nan

                    # grass被误分类为vegetation的比例
                    if grass_true_count > 0:
                        data_row['grass2vegetation'] = score_matrix[grass_id][vegetation_id] / grass_true_count
                    else:
                        data_row['grass2vegetation'] = np.nan

                    # background被误分类为grass的比例
                    background_true_count = np.sum(score_matrix[background_id, :])
                    if background_true_count > 0:
                        data_row['background2grass'] = score_matrix[background_id][grass_id] / background_true_count
                    else:
                        data_row['background2grass'] = np.nan

                    # vegetation被误分类为grass的比例
                    vegetation_true_count = np.sum(score_matrix[vegetation_id, :])
                    if vegetation_true_count > 0:
                        data_row['vegetation2grass'] = score_matrix[vegetation_id][grass_id] / vegetation_true_count
                    else:
                        data_row['vegetation2grass'] = np.nan

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
    pred_folder = r"H:\TempProcess\20251220数据传输\2合并标签"
    label_version_pred ="label05_V1_pred"
    # 注意，内部会默认转ply风格的字典，不需要外部添加（待确定）
    generate_scene_group_csv(output_csv, data_folder=label_folder,pred_folder=pred_folder,label_version_pred=label_version_pred)

