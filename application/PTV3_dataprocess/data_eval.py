'''
评估训练数据质量
'''
import copy
import os
import sys
import torch
import path.path_process as path_process
import points.points_eval as points_eval
import data_dict

# 将字符串控制台输出并写出到文件
def PW(text, file_path):
    print(text)
    with open(file_path, 'a') as f:
        f.write(text + '\n')

# 1 评估训练数据质量
def eval_train_data(data_folder, eval_file, data_dict):  #, data_dict, label_version
    '''
    评估训练数据：
        数据输出指标
        可能问题分析
        数据需要符合那些指标来保证
    参考信息： 场景分布、训练验证集分布、点云比例分布
    功能：
        1、当前只处理pth评估。 TODO:兼容 pth、ply文件
        2、    TODO： 当前写死类别，后续开放

    '''
    str_start = """
当前对点云分割数据进行评估，总共标注类别有5类，
    """
    PW (str_start, eval_file)
    print("当前处理的文件为 pth 数据")
    all_info = dict()   # 详细信息
    # 汇总信息
    sum_sub = {
            '总点数量':0,
            '背景':{'平坦':0, '斜坡':0, '总数':0, },
            '建筑':{'高楼':0, '民房':0, '临建':0, '总数':0 },
            '车辆':{'普通汽车':0, '工程车辆':0, '总数':0 },
            '高植被':{'总数':0},
            '低植被':{'总数':0}
            }
    sum_info = {
        'train': copy.deepcopy(sum_sub),
        'val': copy.deepcopy(sum_sub),
        'test': copy.deepcopy(sum_sub)
    }
    # 1 点云信息
    sub_folders = ['train', 'val', 'test']
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(data_folder, sub_folder)
        if not os.path.exists(sub_folder_path):
            continue
        all_info[sub_folder] = dict()
        filelist = path_process.get_files_by_format(sub_folder_path, formats=['.pth'], return_full_path=False)
        all_info[sub_folder]['file_num'] = len(filelist)
        all_info[sub_folder]['scene']= dict()
        for file in filelist:
            scenei_info = dict()
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(sub_folder_path, file)
            PW(f"处理文件: {file_name}",eval_file)
            pth_info = torch.load(file_path)
            label = pth_info['semantic_gt20']
            scenei_info['pts_num'] = label.size
            scenei_info['labels'] = points_eval.label_rate(label)
            all_info[sub_folder]['scene'][file_name] = scenei_info
            # 计算子类



    # 2 输出
    PW("数据质量评估结果：", eval_file)
    # 2.1 输出算量
    PW("总数据集评估：", eval_file)







        # 2 评估推理结果问题
# 混淆矩阵



# 3 评估模型问题


if __name__ == '__main__':
    data_folder = r'/media/xuek/Data210/数据集/临时测试区/20251216训练/5_pth'
    eval_file = r'/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/eval_file_2025.12.18.txt'


    eval_train_data(data_folder, eval_file)