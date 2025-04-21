import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import glob
import numpy as np
import open3d as o3d
import data_parse_3d
import path_process


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


# 标签评估
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


# 文件评估
def file_eval(file1,file2,classNum):
    '''
    评估两个文件，支持的格式参考 data_parse_3d.parse_3d_cloud_file
    '''
    _, _, labels1 = data_parse_3d.parse_3d_cloud_file(file1)
    _, _, labels2 = data_parse_3d.parse_3d_cloud_file(file2)
    if labels1 == None:
        raise ValueError(f'文件1标签解析错误：{file1}')
    if labels2 == None:
        raise ValueError(f'文件2标签解析错误：{file1}')
    scoreMatrix_i = label_eval(labels1, labels2, classNum)

def folder_eval(folder1, folder2, classNum=2):
    '''
    两个文件夹对应文件对比评估，并最后返回所有结果。
    注意：
        1）默认两个文件夹中比较的文件同名！
        2）在folder2 中找 folder1中同名文件
        3) 当前不支持多层文件夹，需要的话重新开发一个功能：【multifolder_eval】
    '''
    # 1 返回文件夹列表
    formats = ['.ply', '.pcd']
    file_list1 = path_process.get_files_by_format(folder1, formats)
    file_list2 = path_process.get_files_by_format(folder2, formats)
    # 2 计算IOU
    scoreMatrix = np.zeros((classNum, classNum), dtype='int')  # 记录精度评价二维矩阵[4][4] ,【真值标签】【推理标签】
    for name_ext2 in file_list2:
        name, ext2 = os.path.splitext(name_ext2)
        file_ext2 = os.path.join(folder2, name_ext2)
        item_list = path_process.find_str_in_strlist(name, file_list1)
        if len(item_list)>1:
            print(f"文件匹配到多个！")
        elif len(item_list)==0:
            raise FileNotFoundError(f'没有匹配文件{name_ext2}')
        file_ext1 = os.path.join(folder1, file_list1[item_list[0]])
        coords1, colors1, labels1 = data_parse_3d.parse_3d_cloud_file(file_ext1)
        coords2, colors2, labels2 = data_parse_3d.parse_3d_cloud_file(file_ext2)
        # labels1 = labels1 / 3  # 注意label是否对应正确！
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

if __name__ == '__main__':
    if False:
        folder = r'/home/xuek/桌面/TestData/input/staticmap_test/IOU'
        folder_out = r'/home/xuek/桌面/PTV3_Versions/ptv3_raser_cpu/ptv3_deploy/scripts/output/IOU/04073'
        folder_eval(folder, folder_out, classNum=2)
