'''
批量评估两个文件夹下的PLY文件对比，并输出CSV格式结果
包含每个文件各类别的recall、precision、iou指标
'''
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# 添加必要的路径
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import points.points_eval as points_eval
import points.points_io as points_io
import path.path_process as path_process


def eval_ply_folder_comparison_csv(folder1, folder2, label_name1, label_name2, output_csv=None, class_num=5, formats=['.ply']):
    '''
    批量比较两个文件夹下同名PLY文件的标签差异，并输出CSV格式结果

    参数:
        folder1: 第一个文件夹路径
        folder2: 第二个文件夹路径
        label_name1: 第一个文件夹文件的标签字段名
        label_name2: 第二个文件夹文件的标签字段名
        output_csv: 输出CSV文件路径（可选）
        class_num: 类别数量（默认为5）
        formats: 支持的文件格式列表

    返回:
        DataFrame包含所有文件的评估结果
    '''
    print(f"\n开始批量评估两个文件夹的PLY文件:")
    print(f"文件夹1: {folder1}, 标签字段: {label_name1}")
    print(f"文件夹2: {folder2}, 标签字段: {label_name2}")

    # 获取两个文件夹的文件列表
    files1 = path_process.get_files_by_format(folder1, formats, return_full_path=False)
    files2 = path_process.get_files_by_format(folder2, formats, return_full_path=True)

    # 创建文件名到完整路径的映射
    file2_path_map = {}
    for file_path in files2:
        filename = os.path.basename(file_path)
        file2_path_map[filename] = file_path

    # 类别名称映射
    label_names = {
        0: "背景类",
        1: "建筑类",
        2: "车辆类",
        3: "高植被类",
        4: "低植被类"
    }

    # 存储所有文件的评估结果
    all_results = []

    # 初始化累计混淆矩阵
    total_confusion_matrix = np.zeros((class_num, class_num))
    valid_files = 0
    failed_files = 0

    # 遍历文件夹1中的文件
    for filename in files1:
        # 尝试多种匹配方式
        file2_path = None
        matched_file2_name = None

        # 获取文件夹1文件的基本名称（不带扩展名）
        base_name = os.path.splitext(filename)[0]

        # 在文件夹2的所有文件中搜索匹配的文件
        for file2_name, file2_full_path in file2_path_map.items():
            # 如果文件夹2的文件名包含文件夹1的基本名称，则认为匹配
            if base_name in file2_name:
                file2_path = file2_full_path
                matched_file2_name = file2_name
                print(f"  找到匹配文件: {filename} <-> {file2_name}")
                break

        if file2_path:
            file1_path = os.path.join(folder1, filename)

            try:
                print(f"\n处理文件: {filename}")

                # 读取第一个文件的标签
                points_dict1 = points_io.parse_cloud_to_dict(file1_path)
                if label_name1 not in points_dict1:
                    print(f"  警告：文件1没有找到标签字段 {label_name1}，跳过")
                    continue
                labels1 = points_dict1[label_name1]

                # 读取第二个文件的标签
                points_dict2 = points_io.parse_cloud_to_dict(file2_path)
                if label_name2 not in points_dict2:
                    print(f"  警告：文件2没有找到标签字段 {label_name2}，跳过")
                    continue
                labels2 = points_dict2[label_name2]

                # 检查标签数量是否一致
                if len(labels1) != len(labels2):
                    print(f"  警告：标签数量不一致！文件1: {len(labels1)}, 文件2: {len(labels2)}")
                    min_len = min(len(labels1), len(labels2))
                    labels1 = labels1[:min_len]
                    labels2 = labels2[:min_len]
                    print(f"  截取前 {min_len} 个点进行比较")

                # 确保标签是numpy数组
                if not isinstance(labels1, np.ndarray):
                    labels1 = np.array(labels1)
                if not isinstance(labels2, np.ndarray):
                    labels2 = np.array(labels2)

                # 计算混淆矩阵
                confusion_matrix = points_eval.matrix_eval(labels1, labels2, class_num)

                # 计算各项指标
                recall, precision, iou = points_eval.eval_result(confusion_matrix, class_num)

                # 创建文件结果记录
                file_result = {
                    '数据名': os.path.splitext(filename)[0],  # 使用不带扩展名的文件名
                    '点云数量': len(labels1),
                    '文件1路径': file1_path,
                    '文件2路径': file2_path
                }

                # 添加各类别的指标
                for i in range(class_num):
                    class_name = label_names.get(i, f"类别{i}")
                    file_result[f'{class_name}_Recall'] = recall[i]
                    file_result[f'{class_name}_Precision'] = precision[i]
                    file_result[f'{class_name}_IoU'] = iou[i]

                # 添加平均指标
                file_result['平均_Recall'] = np.mean(recall)
                file_result['平均_Precision'] = np.mean(precision)
                file_result['平均_IoU'] = np.mean(iou)

                all_results.append(file_result)

                # 累计混淆矩阵用于汇总统计
                total_confusion_matrix += confusion_matrix
                valid_files += 1

                # 输出单文件结果
                print(f"  平均指标: Recall={np.mean(recall):.2%}, Precision={np.mean(precision):.2%}, IoU={np.mean(iou):.2%}")

            except Exception as e:
                print(f"  错误：处理文件 {filename} 时出错 - {str(e)}")
                failed_files += 1
        else:
            print(f"警告：文件夹2中没有找到文件 {filename}")
            failed_files += 1

    # 转换为DataFrame
    df = pd.DataFrame(all_results)

    # 添加汇总统计行
    if valid_files > 0:
        # 基于累计混淆矩阵计算点云级别的平均指标
        total_recall, total_precision, total_iou = points_eval.eval_result(total_confusion_matrix, class_num)

        # 创建汇总行
        summary_row = {
            '数据名': '汇总统计',
            '点云数量': df['点云数量'].sum() if len(df) > 0 else 0,
            '文件1路径': '',
            '文件2路径': ''
        }

        # 添加各类别的平均指标
        for i in range(class_num):
            class_name = label_names.get(i, f"类别{i}")
            summary_row[f'{class_name}_Recall'] = total_recall[i]
            summary_row[f'{class_name}_Precision'] = total_precision[i]
            summary_row[f'{class_name}_IoU'] = total_iou[i]

        # 添加平均指标
        summary_row['平均_Recall'] = np.mean(total_recall)
        summary_row['平均_Precision'] = np.mean(total_precision)
        summary_row['平均_IoU'] = np.mean(total_iou)

        # 将汇总行添加到DataFrame
        df_summary = pd.DataFrame([summary_row])
        df = pd.concat([df, df_summary], ignore_index=True)

    # 输出汇总结果
    print("\n" + "="*80)
    print("批量评估汇总:")
    print(f"成功对比文件数: {valid_files}")
    print(f"失败文件数: {failed_files}")

    # 保存到CSV
    if output_csv and len(df) > 0:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n评估结果已保存到CSV文件: {output_csv}")

    # 打印部分结果（如果数据量不大）
    if len(df) > 0 and len(df) <= 20:
        print("\n评估结果:")
        print("-" * 80)
        # 选择主要列显示
        display_cols = ['数据名', '点云数量', '平均_Recall', '平均_Precision', '平均_IoU']
        for i in range(class_num):
            class_name = label_names.get(i, f"类别{i}")
            display_cols.extend([f'{class_name}_IoU'])

        if all(col in df.columns for col in display_cols):
            print(df[display_cols].to_string(index=False))

    return df


def eval_ply_folder_comparison_csv_simple(folder1, folder2, label_name1, label_name2, output_csv=None, class_num=5, formats=['.ply']):
    '''
    简化版本的批量评估函数，不依赖外部模块

    参数:
        folder1: 第一个文件夹路径
        folder2: 第二个文件夹路径
        label_name1: 第一个文件夹文件的标签字段名
        label_name2: 第二个文件夹文件的标签字段名
        output_csv: 输出CSV文件路径（可选）
        class_num: 类别数量（默认为5）
        formats: 支持的文件格式列表

    返回:
        DataFrame包含所有文件的评估结果
    '''
    import open3d as o3d

    print(f"\n开始批量评估两个文件夹的PLY文件:")
    print(f"文件夹1: {folder1}, 标签字段: {label_name1}")
    print(f"文件夹2: {folder2}, 标签字段: {label_name2}")

    # 获取两个文件夹的文件列表
    files1 = get_files_by_format(folder1, formats)
    files2 = get_files_by_format(folder2, formats)

    # 创建文件名到完整路径的映射
    file2_path_map = {}
    for file_path in files2:
        filename = os.path.basename(file_path)
        file2_path_map[filename] = file_path

    # 类别名称映射
    label_names = {
        0: "背景类",
        1: "建筑类",
        2: "车辆类",
        3: "高植被类",
        4: "低植被类"
    }

    # 存储所有文件的评估结果
    all_results = []

    # 初始化累计混淆矩阵
    total_confusion_matrix = np.zeros((class_num, class_num))
    valid_files = 0
    failed_files = 0

    # 遍历文件夹1中的文件
    for filename in files1:
        # 尝试多种匹配方式
        file2_path = None
        matched_file2_name = None

        # 获取文件夹1文件的基本名称（不带扩展名）
        base_name = os.path.splitext(filename)[0]

        # 在文件夹2的所有文件中搜索匹配的文件
        for file2_name, file2_full_path in file2_path_map.items():
            # 如果文件夹2的文件名包含文件夹1的基本名称，则认为匹配
            if base_name in file2_name:
                file2_path = file2_full_path
                matched_file2_name = file2_name
                print(f"  找到匹配文件: {filename} <-> {file2_name}")
                break

        if file2_path:
            file1_path = os.path.join(folder1, filename)

            try:
                print(f"\n处理文件: {filename}")

                # 读取PLY文件并获取标签
                labels1 = read_labels_from_ply(file1_path, label_name1)
                labels2 = read_labels_from_ply(file2_path, label_name2)

                if labels1 is None or labels2 is None:
                    print(f"  警告：无法读取标签，跳过")
                    continue

                # 检查标签数量是否一致
                if len(labels1) != len(labels2):
                    print(f"  警告：标签数量不一致！文件1: {len(labels1)}, 文件2: {len(labels2)}")
                    min_len = min(len(labels1), len(labels2))
                    labels1 = labels1[:min_len]
                    labels2 = labels2[:min_len]
                    print(f"  截取前 {min_len} 个点进行比较")

                # 计算混淆矩阵
                confusion_matrix = compute_confusion_matrix(labels1, labels2, class_num)

                # 计算各项指标
                recall, precision, iou = compute_metrics(confusion_matrix, class_num)

                # 创建文件结果记录
                file_result = {
                    '数据名': os.path.splitext(filename)[0],  # 使用不带扩展名的文件名
                    '点云数量': len(labels1),
                    '文件1路径': file1_path,
                    '文件2路径': file2_path
                }

                # 添加各类别的指标
                for i in range(class_num):
                    class_name = label_names.get(i, f"类别{i}")
                    file_result[f'{class_name}_Recall'] = recall[i]
                    file_result[f'{class_name}_Precision'] = precision[i]
                    file_result[f'{class_name}_IoU'] = iou[i]

                # 添加平均指标
                file_result['平均_Recall'] = np.mean(recall)
                file_result['平均_Precision'] = np.mean(precision)
                file_result['平均_IoU'] = np.mean(iou)

                all_results.append(file_result)

                # 累计混淆矩阵用于汇总统计
                total_confusion_matrix += confusion_matrix
                valid_files += 1

                # 输出单文件结果
                print(f"  平均指标: Recall={np.mean(recall):.2%}, Precision={np.mean(precision):.2%}, IoU={np.mean(iou):.2%}")

            except Exception as e:
                print(f"  错误：处理文件 {filename} 时出错 - {str(e)}")
                failed_files += 1
        else:
            print(f"警告：文件夹2中没有找到文件 {filename}")
            failed_files += 1

    # 转换为DataFrame
    df = pd.DataFrame(all_results)

    # 添加汇总统计行
    if valid_files > 0:
        # 基于累计混淆矩阵计算点云级别的平均指标
        total_recall, total_precision, total_iou = compute_metrics(total_confusion_matrix, class_num)

        # 创建汇总行
        summary_row = {
            '数据名': '汇总统计',
            '点云数量': df['点云数量'].sum() if len(df) > 0 else 0,
            '文件1路径': '',
            '文件2路径': ''
        }

        # 添加各类别的平均指标
        for i in range(class_num):
            class_name = label_names.get(i, f"类别{i}")
            summary_row[f'{class_name}_Recall'] = total_recall[i]
            summary_row[f'{class_name}_Precision'] = total_precision[i]
            summary_row[f'{class_name}_IoU'] = total_iou[i]

        # 添加平均指标
        summary_row['平均_Recall'] = np.mean(total_recall)
        summary_row['平均_Precision'] = np.mean(total_precision)
        summary_row['平均_IoU'] = np.mean(total_iou)

        # 将汇总行添加到DataFrame
        df_summary = pd.DataFrame([summary_row])
        df = pd.concat([df, df_summary], ignore_index=True)

    # 输出汇总结果
    print("\n" + "="*80)
    print("批量评估汇总:")
    print(f"成功对比文件数: {valid_files}")
    print(f"失败文件数: {failed_files}")

    # 保存到CSV
    if output_csv and len(df) > 0:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n评估结果已保存到CSV文件: {output_csv}")

    # 打印部分结果（如果数据量不大）
    if len(df) > 0 and len(df) <= 20:
        print("\n评估结果:")
        print("-" * 80)
        # 选择主要列显示
        display_cols = ['数据名', '点云数量', '平均_Recall', '平均_Precision', '平均_IoU']
        for i in range(class_num):
            class_name = label_names.get(i, f"类别{i}")
            display_cols.append(f'{class_name}_IoU')

        # 只显示存在的列
        existing_cols = [col for col in display_cols if col in df.columns]
        if existing_cols:
            print(df[existing_cols].to_string(index=False))

    return df


def get_files_by_format(folder, formats):
    """
    获取文件夹中指定格式的文件
    """
    if not os.path.exists(folder):
        return []

    files = []
    for file in os.listdir(folder):
        if any(file.lower().endswith(fmt.lower()) for fmt in formats):
            files.append(file)

    return files


def read_labels_from_ply(file_path, label_field):
    """
    从PLY文件读取标签
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)

        # 尝试不同的方法获取标签
        if hasattr(pcd, label_field):
            labels = getattr(pcd, label_field)
        else:
            # 尝试从点云的其他属性中获取
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return None

            # 这里需要根据实际的PLY文件结构来获取标签
            # 简化处理：假设没有找到标签字段
            print(f"  警告：未找到标签字段 {label_field}")
            return None

        return np.array(labels)
    except Exception as e:
        print(f"  读取PLY文件出错: {str(e)}")
        return None


def compute_confusion_matrix(label1, label2, class_num):
    """
    计算混淆矩阵
    """
    matrix = np.zeros((class_num, class_num), dtype=int)
    for i in range(len(label1)):
        if label1[i] < class_num and label2[i] < class_num:
            matrix[label1[i], label2[i]] += 1
    return matrix


def compute_metrics(confusion_matrix, class_num):
    """
    计算召回率、精确率、IoU
    """
    # 真正例
    tp = np.diag(confusion_matrix)

    # 召回率：TP / (TP + FN)
    fn = np.sum(confusion_matrix, axis=1) - tp
    recall = tp / (tp + fn + 1e-7)

    # 精确率：TP / (TP + FP)
    fp = np.sum(confusion_matrix, axis=0) - tp
    precision = tp / (tp + fp + 1e-7)

    # IoU：TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn + 1e-7)

    return recall, precision, iou


if __name__ == '__main__':
    # 示例用法
    print("PLY文件批量评估CSV输出")
    print("="*50)

    # 获取用户输入
    folder1 = input("请输入第一个文件夹路径（真实标签）: ") or '/media/xuek/Data210/数据集/训练集/重建数据_动态维护_ply'
    folder2 = input("请输入第二个文件夹路径（预测标签）: ") or '/media/xuek/Data210/数据集/临时测试区/20251221版本/推理结果'
    label_name1 = input("请输入第一个文件夹的标签字段名: ") or 'label_label05_V1'
    label_name2 = input("请输入第二个文件夹的标签字段名: ") or 'class_class'
    output_csv = input("请输入输出CSV文件路径: ") or '/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/PLY评估结果.csv'
    class_num = int(input("请输入类别数量 (默认5): ") or "5")

    # 执行评估
    try:
        # 尝试使用完整版本
        df = eval_ply_folder_comparison_csv(folder1, folder2, label_name1, label_name2, output_csv, class_num)
    except:
        # 如果失败，使用简化版本
        print("使用简化版本进行评估...")
        df = eval_ply_folder_comparison_csv_simple(folder1, folder2, label_name1, label_name2, output_csv, class_num)

    if len(df) > 0:
        print("\n评估完成！")
        print(f"共评估了 {len(df)-1} 个文件")  # 减1是因为包含汇总行