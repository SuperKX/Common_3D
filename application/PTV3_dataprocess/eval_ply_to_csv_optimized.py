'''
批量评估两个文件夹下的PLY文件对比，并输出CSV格式结果（优化版）
包含每个文件各类别的recall、precision、iou指标
特别增加低矮植被与背景类、高植被类的混淆程度分析
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
    批量比较两个文件夹下同名PLY文件的标签差异，并输出CSV格式结果（优化版）

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

                # 计算混淆程度（class=4与class=0和class=3的混淆）
                # 低矮植被(4)被误分为背景(0)的比例
                low_veg_to_bg = confusion_matrix[4, 0] / (confusion_matrix[4].sum() + 1e-7)
                # 低矮植被(4)被误分为高植被(3)的比例
                low_veg_to_high_veg = confusion_matrix[4, 3] / (confusion_matrix[4].sum() + 1e-7)
                # 背景(0)被误分为低矮植被(4)的比例
                bg_to_low_veg = confusion_matrix[0, 4] / (confusion_matrix[0].sum() + 1e-7)
                # 高植被(3)被误分为低矮植被(4)的比例
                high_veg_to_low_veg = confusion_matrix[3, 4] / (confusion_matrix[3].sum() + 1e-7)

                # 创建文件结果记录
                file_result = {
                    '数据名': os.path.splitext(filename)[0],  # 使用不带扩展名的文件名
                    '低矮植被->背景混淆': low_veg_to_bg,
                    '低矮植被->高植被混淆': low_veg_to_high_veg,
                    '背景->低矮植被混淆': bg_to_low_veg,
                    '高植被->低矮植被混淆': high_veg_to_low_veg
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
                print(f"  低矮植被混淆: ->背景={low_veg_to_bg:.2%}, ->高植被={low_veg_to_high_veg:.2%}")

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

        # 计算汇总的混淆程度
        total_low_veg_to_bg = total_confusion_matrix[4, 0] / (total_confusion_matrix[4].sum() + 1e-7)
        total_low_veg_to_high_veg = total_confusion_matrix[4, 3] / (total_confusion_matrix[4].sum() + 1e-7)
        total_bg_to_low_veg = total_confusion_matrix[0, 4] / (total_confusion_matrix[0].sum() + 1e-7)
        total_high_veg_to_low_veg = total_confusion_matrix[3, 4] / (total_confusion_matrix[3].sum() + 1e-7)

        # 创建汇总行
        summary_row = {
            '数据名': '汇总统计',
            '低矮植被->背景混淆': total_low_veg_to_bg,
            '低矮植被->高植被混淆': total_low_veg_to_high_veg,
            '背景->低矮植被混淆': total_bg_to_low_veg,
            '高植被->低矮植被混淆': total_high_veg_to_low_veg
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

    if valid_files > 0:
        print("\n低矮植被混淆分析:")
        print(f"  低矮植被->背景混淆: {total_low_veg_to_bg:.2%}")
        print(f"  低矮植被->高植被混淆: {total_low_veg_to_high_veg:.2%}")
        print(f"  背景->低矮植被混淆: {total_bg_to_low_veg:.2%}")
        print(f"  高植被->低矮植被混淆: {total_high_veg_to_low_veg:.2%}")

    # 保存到CSV
    if output_csv and len(df) > 0:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n评估结果已保存到CSV文件: {output_csv}")

    # 打印部分结果（如果数据量不大）
    if len(df) > 0 and len(df) <= 20:
        print("\n评估结果:")
        print("-" * 80)
        # 选择主要列显示
        display_cols = ['数据名', '低矮植被->背景混淆', '低矮植被->高植被混淆', '平均_Recall', '平均_Precision', '平均_IoU']
        for i in range(class_num):
            class_name = label_names.get(i, f"类别{i}")
            display_cols.append(f'{class_name}_IoU')

        if all(col in df.columns for col in display_cols):
            print(df[display_cols].to_string(index=False))

    return df


# 示例用法
if __name__ == '__main__':
    # 设置参数
    folder1 = '/media/xuek/Data210/数据集/训练集/重建数据_动态维护_ply'
    folder2 = '/media/xuek/Data210/数据集/临时测试区/20251221版本/推理结果'
    label_name1 = 'label_label05_V1'
    label_name2 = 'class_class'
    output_csv = '/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/PLY评估结果_optimized.csv'
    class_num = 5

    # 执行评估
    df = eval_ply_folder_comparison_csv(folder1, folder2, label_name1, label_name2, output_csv, class_num)

    if len(df) > 0:
        print("\n评估完成！")
        print(f"共评估了 {len(df)-1} 个文件")  # 减1是因为包含汇总行