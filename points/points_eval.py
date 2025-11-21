'''
    点云的标签修改，对比，映射等。
    label_rate          计算标签中各类别比例。
    file_label_info     计算文件中标签类别比例（支持映射后）
    label_change        根据字典，修改标签值
    matrix_eval         计算两个标签的混淆矩阵
    eval_result         根据评价矩阵，计算准确率、召回率、iou
    label_eval          输入两个label,写出评价结果，并返回混淆矩阵
    labels_file_eval    评估1个文件的两组标签
    labels_file_folder_eval  批量比较两个文件夹下的标签
    files_eval          评估两个文件，支持的格式参考 data_parse_3d.parse_3d_cloud_file
    folder_eval         两个文件夹对应文件对比评估，并最后返回所有结果
'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import glob
import numpy as np
import open3d as o3d
import points.points_io as pts_io
import path.path_process as path_process
import application.PTV3_dataprocess.label_dict as label_dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from path import path_process as pth_process

def label_rate(labels):
    '''
    输入指定标签，给出各标签比例等信息。
    '''
    unique_orig, counts_orig = np.unique(labels, return_counts=True)
    print(f"  点云数量: {labels.size}")
    for u, c in zip(unique_orig, counts_orig):
        print(f'    标签：{u} ，数量：{c} ， 比例：{c / labels.size:.2%}')

def file_label_info(file, label_name,label_dict={}):
    '''
    查看标签信息,可计算变化后的信息。
    参数:
        file: 点云文件路径
        label_name: 标签名称
        label_dict: 标签映射字典，用于标签转换，默认为空字典
    '''
    print(f'【文件】：{os.path.basename(file)} 标签信息：')
    points_dict = pts_io.parse_cloud_to_dict(file)
    labels = points_dict[label_name]
    if label_dict != {}:  # 映射字典不为空
        labels = label_change(labels, label_dict)
    label_rate(labels)


# 标签映射
def label_change(labels, label_dict):
    '''
    TODO： preprocess.py中验证通过后，调用此处接口。
    根据字典指引，修改点云标签
    params:
        labels      原始标签 ndarray(n，)
        label_dict  记录新旧点云映射，{新标签1：[旧标签1，就标签2, ...], },ex: label_dict={0:[0,1],1:[2,3]}；未修改标签则置为0
    return：
        labels_new  返回新标签
    '''
    # 判空
    if len(label_dict) == 0:
        print("【注意】：没有发生标签修改！")
        return labels

    # 修改标签
    labels_new = np.zeros((len(labels)), dtype=np.int8)  # 构造新的标签ndarray（N，）
    changed_label_num = 0
    for new_label,old_label in label_dict.items():  # 逐个新标签
        mask = np.isin(labels, old_label)
        changed_label_num += mask.sum()
        labels_new[mask] = new_label
    if changed_label_num < labels.size:
        print(f'【注意】有 {labels.size - changed_label_num} 点 未修改标签！')

    return labels_new

# 计算混淆矩阵
def matrix_eval(label,pred, classNum):
    '''
    根据标签、推理值，估计评价矩阵 （行：标签，列：推理值）
    label       标签值，ndarray（n,）
    pred        推理值，ndarray（n,）
    classNum    类别数量，（标签默认范围[0,classname))
    '''
    scoreMatrix = np.zeros((classNum, classNum), dtype='int')  # 记录每个数据的评价矩阵
    # 构造评价矩阵
    listnew = label * classNum + pred  # 将两个列表映射为1维列表，保证不同真值及推理值都对应一个新的值，[4][4]->[0-150]
    # score = listnew == 0
    for cls_true in range(classNum):  # 真值标签
        for cls_predict in range(classNum):  # 推理标签
            scoreMatrix[cls_true][cls_predict] += np.sum(
                listnew == cls_true * classNum + cls_predict)  # 如所有值为12的点，表示为[标签4][推理0]对应的点数量
    return scoreMatrix

# 根据混淆矩阵,返回评价结果
def eval_result(scoreMatrix, classNum):
    '''
    根据评价矩阵，计算准确率、召回率、iou
    return
        score_rec   recall
        score_pre   precision
        score_iou   iou
    '''
    list_TP = np.array([scoreMatrix[i][i] for i in range(classNum)])  # TP
    list_rec = np.sum(scoreMatrix, axis=1)  # 召回
    list_pre = np.sum(scoreMatrix, axis=0)  # 准确
    list_u = list_rec + list_pre - list_TP  # 并集
    score_rec = list_TP / (list_rec + 1e-7)
    score_pre = list_TP / (list_pre + 1e-7)
    score_iou = list_TP / (list_u + 1e-7)
    return score_rec, score_pre, score_iou

# 标签比较
def label_eval(label1, label2, classNum=4):
    '''
    输入两个label,写出评价结果，并返回混淆矩阵
    label       标签值，ndarray（n,）
    pred        推理值，ndarray（n,）
    classNum    类别数量，（标签默认范围[0,classname))
    '''
    # 构造评价矩阵
    scoreMatrix_i = matrix_eval(label1, label2, classNum)
    # 评价
    score_rec, score_pre, score_iou = eval_result(scoreMatrix_i, classNum)
    for classi in range(classNum):
        print(
            f'class{classi}: recall {score_rec[classi]:.1%}, precision {score_pre[classi]:.1%}, iou {score_iou[classi]:.1%}')
    return scoreMatrix_i

# 多标签文件比较
def labels_file_eval(file, classNum, label_name1, label_name2, label_dict1={}, label_dict2={}):
    '''
    评估1个文件的两组标签
    paras:
        file           文件地址
        classNum       类别数量
        label_name1    第一组标签名称
        label_name2    第二组标签名称
        label_dict1    第一组标签字典映射
        label_dict2    第二组标签字典映射
    return:
        scoreMatrix_i  评分矩阵
    '''
    # 1 读取文件
    points_dict = pts_io.parse_cloud_to_dict(file)
    if label_name1 not in points_dict:
        raise ValueError(f'文件1没有{label_name1}标签：{file}')
    if label_name2 not in points_dict:
        raise ValueError(f'文件2没有{label_name2}标签：{file}')

    # 2 读取标签
    labels1 = points_dict[label_name1]
    labels2 = points_dict[label_name2]

    # 3 标签重映射
    labels1 = label_change(labels1, label_dict1)
    labels2 = label_change(labels2, label_dict2)

    # 比较两个label
    scoreMatrix_i = label_eval(labels1, labels2, classNum)
    return scoreMatrix_i

# 多标签文件批量比较
def labels_file_folder_eval(folder, classNum, label_name1, label_name2, label_dict1={}, label_dict2={}):
    '''
    批量比较两个文件夹下的标签
    paras:
        folder         文件夹
        classNum        对比的类别数量（当前显式传入）
        label_name1     文件1标签名称
        label_name2     文件2标签名称
        label_dict1     文件1字典映射
        label_dict2     文件2字典映射
    return
        scoreMatrix_i   评分矩阵
    '''
    formats = ['.ply', '.pcd']
    file_list = pth_process.get_files_by_format(folder, formats)
    if not os.path.exists(folder):
        raise ValueError(f'地址找不到文件夹：{folder}')
    if len(file_list) == 0:
        raise ValueError(f'地址找不到目标格式的文件：{folder}')

    # 2 初始化总评分矩阵
    total_scoreMatrix = np.zeros((classNum, classNum), dtype='int')

    # 3 遍历文件进行评估
    for file_name in file_list:
        file_path = os.path.join(folder, file_name)
        try:
            # 使用labels_file_eval函数评估单个文件
            scoreMatrix_i = labels_file_eval(file_path, classNum, label_name1, label_name2, label_dict1, label_dict2)
            total_scoreMatrix += scoreMatrix_i
            print(f"已完成文件评估: {file_name}")
        except Exception as e:
            print(f"文件 {file_name} 评估失败: {e}")
            continue

    # 4 计算总体评估结果
    score_rec, score_pre, score_iou = eval_result(total_scoreMatrix, classNum)
    print(f"\n文件夹 {folder} 总体评估结果:")
    for classi in range(classNum):
        print(
            f'class{classi}: recall {score_rec[classi]:.1%}, precision {score_pre[classi]:.1%}, iou {score_iou[classi]:.1%}')
    print(f'mRecall {score_rec.mean():.1%}, mPrecision {score_pre.mean():.1%}, mIOU {score_iou.mean():.1%}')

    return total_scoreMatrix




# 俩文件比较
def files_eval(file1, file2, classNum, label_dict1={}, label_dict2={}):
    '''
    评估两个文件，支持的格式参考 data_parse_3d.parse_3d_cloud_file
    paras:
        file1           文件1地址
        file2           文件2地址
        label_dict1     文件1字典映射
        label_dict2     文件2字典映射
        classNum        对比的类别数量（当前显式传入）
    return
        scoreMatrix_i   评分矩阵
    '''
    _, _, labels1 = pts_io.parse_3d_cloud_file(file1)
    _, _, labels2 = pts_io.parse_3d_cloud_file(file2)
    if not isinstance(labels1,np.ndarray):
        raise ValueError(f'文件1标签解析错误：{file1}')
    if not isinstance(labels2, np.ndarray):
        raise ValueError(f'文件2标签解析错误：{file2}')
    # 3 标签重映射
    labels1 = label_change(labels1, label_dict1)
    labels2 = label_change(labels2, label_dict2)

    # 比较两个label
    scoreMatrix_i = label_eval(labels1, labels2, classNum)
    return scoreMatrix_i

# 俩文件夹比较
def folder_eval(folder1, folder2, classNum=2,label_dict1={},label_dict2={}):
    '''
    两个文件夹对应文件对比评估，并最后返回所有结果。
    注意：
        0）兼容混合多种格式匹配，但是匹配文件格式相同（避免同名不同格式的文件为不同文件。如p1.pcd和P2.pcd）
        1）默认两个文件夹中比较的文件同名！
        2）在folder2 中找 folder1中同名文件
        3) 当前不支持多层文件夹，需要的话重新开发一个功能：【multifolder_eval】-可参考
        4）标签若没对齐，记得修改！！（如车辆标注标签3，推理标签1：labels1 = labels1 / 3 ）
        5) 优化:classnum改为字典:前后类别标签对应关系,如{0:0,1:3}
    '''
    # 1 返回文件夹列表
    formats = ['.ply', '.pcd']
    file_list1 = pth_process.get_files_by_format(folder1, formats)
    file_list2 = pth_process.get_files_by_format(folder2, formats)
    if not os.path.exists(folder1):
        raise ValueError(f'地址找不到文件夹：{folder1}')
    if not os.path.exists(folder2):
        raise ValueError(f'地址找不到文件夹：{folder2}')
    if len(file_list1) == 0:
        raise ValueError(f'地址找不到目标格式的文件：{folder1}')
    if len(file_list2) == 0:
        raise ValueError(f'地址找不到目标格式的文件：{folder2}')
    # 2 计算IOU
    scoreMatrix = np.zeros((classNum, classNum), dtype='int')  # 记录精度评价二维矩阵[4][4] ,【真值标签】【推理标签】
    for name_ext2 in file_list2:
        # name, ext2 = os.path.splitext(name_ext2)
        file_ext2 = os.path.join(folder2, name_ext2)
        item_list = pth_process.find_str_in_strlist(name_ext2, file_list1)
        if len(item_list)>1:
            print(f"文件匹配到多个！")
        elif len(item_list)==0:
            raise FileNotFoundError(f'没有匹配文件{name_ext2}')
        file_ext1 = os.path.join(folder1, file_list1[item_list[0]])
        scoreMatrix_i = files_eval(file_ext1, file_ext2, classNum, label_dict1, label_dict2)
        scoreMatrix += scoreMatrix_i

    # 评估所有场景
    score_rec, score_pre, score_iou = eval_result(scoreMatrix, classNum)
    print(f"\n所有场景结果如下:")
    for classi in range(classNum):
        print(
            f'class{classi}: recall {score_rec[classi]:.1%}, precision {score_pre[classi]:.1%}, iou {score_iou[classi]:.1%}')
    print(f'mRecall {score_rec.mean():.1%}, mPrecision {score_pre.mean():.1%}, mIOU {score_iou.mean():.1%}')

def folder_eval_上一版备份(folder1, folder2, classNum=2):
    '''
    两个文件夹对应文件对比评估，并最后返回所有结果。
    注意：
        0）兼容混合多种格式匹配，但是匹配文件格式相同（避免同名不同格式的文件为不同文件。如p1.pcd和P2.pcd）
        1）默认两个文件夹中比较的文件同名！
        2）在folder2 中找 folder1中同名文件
        3) 当前不支持多层文件夹，需要的话重新开发一个功能：【multifolder_eval】-可参考
        4）标签若没对齐，记得修改！！（如车辆标注标签3，推理标签1：labels1 = labels1 / 3 ）
        5) 优化:classnum改为字典:前后类别标签对应关系,如{0:0,1:3}
    '''
    # 1 返回文件夹列表
    formats = ['.ply', '.pcd']
    file_list1 = pth_process.get_files_by_format(folder1, formats)
    file_list2 = pth_process.get_files_by_format(folder2, formats)
    if not os.path.exists(folder1):
        raise ValueError(f'地址找不到文件夹：{folder1}')
    if not os.path.exists(folder2):
        raise ValueError(f'地址找不到文件夹：{folder2}')
    if len(file_list1) == 0:
        raise ValueError(f'地址找不到目标格式的文件：{folder1}')
    if len(file_list2) == 0:
        raise ValueError(f'地址找不到目标格式的文件：{folder2}')
    # 2 计算IOU
    scoreMatrix = np.zeros((classNum, classNum), dtype='int')  # 记录精度评价二维矩阵[4][4] ,【真值标签】【推理标签】
    for name_ext2 in file_list2:
        # name, ext2 = os.path.splitext(name_ext2)
        file_ext2 = os.path.join(folder2, name_ext2)
        item_list = pth_process.find_str_in_strlist(name_ext2, file_list1)
        if len(item_list)>1:
            print(f"文件匹配到多个！")
        elif len(item_list)==0:
            raise FileNotFoundError(f'没有匹配文件{name_ext2}')
        file_ext1 = os.path.join(folder1, file_list1[item_list[0]])
        coords1, colors1, labels1 = pts_io.parse_3d_cloud_file(file_ext1)
        coords2, colors2, labels2 = pts_io.parse_3d_cloud_file(file_ext2)
        labels1 = labels1 / 3  # 注意label是否对应正确！
        # 比较两个label
        scoreMatrix_i = label_eval(labels1, labels2, classNum=2)
        scoreMatrix += scoreMatrix_i

    # 评估所有场景
    score_rec, score_pre, score_iou = eval_result(scoreMatrix, classNum)
    print(f"\n所有场景结果如下:")
    for classi in range(classNum):
        print(
            f'class{classi}: recall {score_rec[classi]:.1%}, precision {score_pre[classi]:.1%}, iou {score_iou[classi]:.1%}')
    print(f'mRecall {score_rec.mean():.1%}, mPrecision {score_pre.mean():.1%}, mIOU {score_iou.mean():.1%}')

def labeled_data_standard(file, file_output, label_change_dict):
    '''
        将标注数据标准化：
            1、标签名修改
            2、标签名从点云信息中剥离。
        paras:
            file: 输入点云文件路径
            file_output: 输出点云文件路径
            label_change_dict: 标签字段名称映射字典，格式为 {原标签名: 新标签名}
    '''
    points_dict = pts_io.parse_cloud_to_dict(file)
    for key, new_key in label_change_dict.items():
        if key in points_dict:
            points_dict[new_key] = points_dict.pop(key)
    pts_io.write_ply_from_dict(file_output, points_dict, True)


if __name__ == '__main__':
    if True:  # 标签标准化
        if False:  # 单文件
            file = r'J:\DATASET\BIMTwins\版本2025.10.15/01JXY.ply'
            file_output = r'J:\DATASET\BIMTwins\标准数据集/01JXY.ply'
            label_change_dict = {"vertex_class": "label12_V1", "class_class": "label05_V1_predict"}
            labeled_data_standard(file, file_output, label_change_dict)
        else:
            folder = r'J:\DATASET\BIMTwins\版本2025.10.15'
            folder_output = r'J:\DATASET\BIMTwins\标准数据集'
            label_change_dict = {"vertex_class": "label12_V1", "vertex_scalar_class": "label12_V1"}
            file_list = pth_process.get_files_by_format(folder, ['.ply', '.pcd'], return_full_path=False)
            for name_ext in file_list:
                print(f"开始处理: {name_ext}")
                labeled_data_standard(os.path.join(folder, name_ext), os.path.join(folder_output, name_ext), label_change_dict)


    if False:  # 查看标签标签比例
        if False:  # 单文件
            file = r'/media/xuek/Data210/数据集/训练集/重建数据_版本2025.10.15/train/33ZJ.ply'
            label_name = "vertex" + "_" + "class"  # element_name + 属性名
            # file_label_info(file, label_name)
            file_label_info(file, label_name, label_dict=label_dict.label_map_l12v1_to_l05v1)
        else:  # 文件夹遍历
            folder = r'/media/xuek/Data210/数据集/训练集/重建数据_版本2025.10.15/val'
            label_name = "vertex" + "_" + "class"  # element_name + 属性名
            file_list = pth_process.get_files_by_format(folder, ['.ply', '.pcd'], return_full_path=False)
            for name_ext in file_list:
                file_label_info(os.path.join(folder, name_ext), label_name, label_dict=label_dict.label_map_l12v1_to_l05v1)

    if False:  # 对比同文件两个变量
        label_dict_input = label_dict.label_map_l12v1_to_l05v1
        if True:  # 1 单文件
            file = r'/home/xuek/桌面/TestData/临时测试区/重建数据_版本2025.10.15_weight20251113/val/val_34PTY1.ply'
            labels_file_eval(file, classNum=5,
                         label_name1="vertex_class", label_name2="class_class",
                         label_dict1=label_dict_5class, label_dict2={})
        else:  # 2 文件夹
            folder = r'/home/xuek/桌面/TestData/临时测试区/重建数据_版本2025.10.15_weight20251113/val'
            labels_file_folder_eval(folder, classNum=5,
                         label_name1="vertex_class", label_name2="class_class",
                         label_dict1=label_dict_5class, label_dict2={})
    if False:
        # 1）6分类数据处理 ["background", "building", "car", "vegetation", "farmland", "grass"]
        label_dict_input = label_dict.label_map_l12v1_to_l06v1
        file1 = r'/media/xuek/Data210/数据集/训练集/重建数据_版本2025.10.15/val/39PTY2.ply'
        file2 = r'/home/xuek/桌面/TestData/临时测试区/重建数据_版本2025.10.15_val_weight20251029/val_39PTY2.ply'
        file_eval(file1, file2, classNum=6, label_dict1=label_dict_6class, label_dict2={})
        # folder_eval(folder1, folder2, classNum=2, label_dict1={}, label_dict2={})

    if False:  # 批量对比两个文件夹下的文件
        # TODO: 老版本，新版本验证通过可删除
        folder = r'/home/xuek/桌面/TestData/input/staticmap_val/验证集0414-6场景'
        folder_out = r'/home/xuek/桌面/PTV3_Versions/ptv3_raser_cpu/ptv3_deploy/scripts/output/staticmap_test_0618/验证集0414-6场景_20250805验证'
        # folder = r'/home/xuek/桌面/TestData/input/staticmap_test/IOU'
        # folder_out = r'/home/xuek/桌面/PTV3_Versions/ptv3_raser_cpu/ptv3_deploy/scripts/output/IOU/04073'
        folder_eval(folder, folder_out, classNum=2)

    if False:  # 对比同文件中的两组标签
        file = r'/home/xuek/桌面/TestData/临时测试区/out_6class_20251020/11分类-临时测试：农田草地突出_34PTY1.ply'
        one_file_eval(file, )