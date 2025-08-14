import os
import argparse
import glob
import json
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
import path.path_process as path_process

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
        unique_labels, counts = np.unique(label, return_counts=True)
        num = label.shape[0]
        print(f"点云数量: {num}")
        for labeli, counti in zip(unique_labels, counts):
            print(f"    标签: {labeli} ,数量: {counti} ,占比: {counti / num:.2%}")

    return


def preprocess_BimTwins(
        inputPath, output_path, parse_normals=True, sample_grid=0.4,
        class_list=[0,1,2,3,4,5,6,8,11],
        new_class=[0,1,1,2,3,4,1,1,5],
        is_create_fake_label=False,
        val_result_pth=True
):
    '''
    brief:
        预处理：读入原始点云数据，生成降采样点云.pth和上采样映射文件。
        注意：
            1）需要用的标签为class_list，修改后的标签为new_class，二者一一对应（长度相同）
            2）未说明的标签，当作0处理
    paras:
        inputPath           输入原始点云
        output_path         输出地址
        parse_normals       是否输出点云法向
        sample_grid         采样密度
        class_list          需要处理的标签
        new_class           修改后的标签
        is_create_fake_label    如果没有标签，则创造假标签
        val_result_pth      是否验证生成的pth文件
    '''
    # 1、获取类别文件夹（train、val、test），判断输入数据类型
    subFoler = inputPath.split('/')[-2]  # 类别文件夹"train"
    scene_id = inputPath.split('/')[-1].split('.')[0]
    output_subsamp_pth = os.path.join(output_path, subFoler, f"{scene_id}.pth")
    output_proj_pkl = os.path.join(output_path, subFoler, f"{scene_id}.pkl")
    print(f"Processing: {scene_id} in {subFoler}")

    # 2 获取点云信息：坐标、颜色、标签
    try:
        coords, colors, labels = points_io.parse_ply_file(inputPath)
    except ValueError as e:
        raise ValueError(f'发生错误{e}')
    if not isinstance(coords, np.ndarray):
        raise ValueError(f'没有点')
    elif not isinstance(labels, np.ndarray):
        if not is_create_fake_label:
            raise ValueError(f'没有标签，如需构造虚假标签，修改变量“is_create_fake_label”！')
        else:
            print(f"【警告】未找到标签信息，此处将构造虚假标签！")
            labels = np.zeros(len(coords), int)

    # 原始标签统计
    unique_orig, counts_orig = np.unique(labels,return_counts=True)
    print(f'原始标签统计：')
    for u, c in zip( unique_orig, counts_orig):
        print(f'    标签：{u} ，数量：{c} ， 比例：{c / labels.size:.2%}')

    # 3 修改标签
    # 3.1 处理已知标签
    labels_new = np.zeros((len(labels)), int)  # 构造新的标签矩
    rest_num = labels.size  # 记录剩余点数量
    class_mapping = dict(zip(class_list,new_class))

    for classid in class_list:
        mask = (labels == classid)
        id_num = mask.sum()
        if id_num > 0:
            labels_new[mask] = class_mapping[classid]
            rest_num -= id_num
    # 3.2 处理未说明标签
    mask_nouse = ~np.isin(labels, class_list)
    num_nouse = mask_nouse.sum()
    if num_nouse>0:
        print(f'    未涵盖标签点数量: {num_nouse}, 比例是{num_nouse / labels.size:.2%}')
        rest_num -= num_nouse
        if True:
            print(f'    将未包含标签改为0')
            labels_new[mask_nouse] = 0

    if rest_num != 0:
        print(f"错误：有 {rest_num} 个点遗漏的标签！")
    # 新标签统计
    unique_new, counts_new = np.unique(labels_new, return_counts=True)
    print(f'新标签统计：')
    for u, c in zip(unique_new, counts_new ):
        print(f'    标签：{u} ，数量：{c} ， 比例：{c / labels.size:.2%}')

    # # class_list.sort()
    # # 1)未标记标签
    # nouse_idx = ~np.isin(labels, class_list)  # 不在目标标签中的数据标签
    # num_nouse = nouse_idx.sum()
    # if num_nouse != 0:
    #     print(f'    未涵盖标签 点数量 {num_nouse}, 比例是{1.0 * num_nouse / labels.size:.1%}')
    #     rest_num -= num_nouse
    #     if True:
    #         print(f'将未包含标签改为0')
    #         labels_new[nouse_idx] = 0
    # # 2）需要修改标签
    # for newid, classid in enumerate(class_list):
    #     id_num = (labels==classid).sum()
    #     print(f'    类别{classid} 的数量是 {id_num}, 比例是{1.0*id_num/labels.size:.1%}')
    #     rest_num -= id_num
    #     if newid != classid:  # 修改id值为0开头
    #         labels_new[labels == classid] = new_class[newid]
    # if rest_num != 0:
    #     print(f"错误：有 {rest_num} 个遗漏的标签！")
    # # 3）标签验证
    # print(f'新标签统计：')
    # for idx in range(max(new_class)+1):
    #     id_num = (labels_new == idx).sum()
    #     print(f'    类别{idx} 的数量是 {id_num}, 比例是{1.0 * id_num / labels.size:.1%}')

    # 4 使用o3d库，创建法向信息
    pcd = o3d.io.read_point_cloud(inputPath)
    # 法线估计
    radius = 0.5  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    # coordsO3D = np.asarray(pcd.points)[:, :]
    # colorsO3D = np.asarray(pcd.colors)[:, :]*255  # 注意此处的颜色为0～1
    # 生成法向信息
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normalsO3D = np.asarray(pcd.normals)[:, :]
    # o3d.io.write_point_cloud("/home/xuek/桌面/Pointcept/data_pre/test.ply", pcd)  # 文件写出

    # 4 降采样
    pcd_grid = pcd.voxel_down_sample(voxel_size=sample_grid)
    coords_sub = np.asarray(pcd_grid.points)[:, :]
    colors_sub = np.asarray(pcd_grid.colors)[:, :] * 255  # 注意此处的颜色为0～1
    save_dict = dict(coord=coords_sub, color=colors_sub, scene_id=scene_id)
    normals_sub = np.asarray(pcd_grid.normals)[:, :]
    if parse_normals:  # 如果存在法向同样补充
        save_dict["normal"] =normals_sub

    # 5 无法用o3d获取标签，用两次kdtree的方式给subsamp的点云重新写入标签
    kd_tree_org = KDTree(data=coords)  # 原始点构造树
    idx_sub = kd_tree_org.query(coords_sub, k=1)[1]  # 降采样点找到原始数据上对应的索引
    labels_sub = labels_new[idx_sub]  # 还原采样点标签
    # Load segments file
    if subFoler != "test":
        semantic_gt20 = labels_sub.astype(int)
        save_dict["semantic_gt20"] = semantic_gt20  # 生成BIMTwins数据集   #semantic_bt4classv2
    # 写出采样后的点云的 pth文件
    torch.save(save_dict, output_subsamp_pth)
    # # 写出sub.ply
    # plysub_save_path = os.path.join(output_path, subFoler, f"{scene_id}_sub.ply")
    # write_ply(labels_sub, colors_sub, coords_sub, plysub_save_path)

    # 6 创建proj文件
    kd_tree_sub = KDTree(data=coords_sub)  # 采样点构造树
    idx_proj = kd_tree_sub.query(coords, k=1)[1]  # 原始数据找到最近的采样点
    with open(output_proj_pkl,'wb') as f:
        pickle.dump([idx_proj, labels], f)  # 写出proj.pkl文件

    if val_result_pth:
        print(f"验证pth数据标签,",end="")
        val_pth(output_subsamp_pth)

'''
参数说明：
    dataset_root    输入数据（输入时已经按照train\ val\ test\ 分配好）
    output_root     输出数据地址
    parse_normals   是否使用点云法向
'''
# 执行命令：
# 1) python preprocess_bimtwins.py --dataset_root /home/xuek/桌面/RandLAnet/Dataset/grid_0.400 --output_root /home/xuek/桌面/Pointcept/data/bimtwins_new
# 后面代码需要法向，暂不禁止法向 --parse_normals True
# 2) main函数中修改执行
if __name__ == "__main__":

    # 手动改值（此处为默认）
    # 1 激光数据
    # dataset_root = r'/home/xuek/桌面/TestData/input/staticmap_dataset/class2_吊车数据集/class2_吊车数据集4RASER_0812'
    # output_root = r'/home/xuek/桌面/TestData/PTV3_data/class2_diaoche/class2_吊车数据集4RASER_0812'
    # class_list = [ 6]  # 需要修改的标签，不处理的默认为0
    # new_class = [ 1]  # 需要改的对应值
    # 2 重建数据
    dataset_root = r'/home/xuek/桌面/TestData/input/11类标签20250730'
    output_root = r'/home/xuek/桌面/TestData/PTV3_data/recon_class11剔除农田改为6类'
    class_list = [0, 1, 2, 3, 4, 5, 6, 8, 11]
    new_class =  [0, 1, 1, 2, 3, 4, 1, 1, 5]

    # 0 config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=False,
        default=dataset_root,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=False,
        default=output_root,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--parse_normals",
        default=True, type=bool,
        help="Whether parse point normals"
    )
    config = parser.parse_args()

    # 1 数据地址
    # 查找所有ply文件
    sceneList = sorted(glob.glob(config.dataset_root + "/*/*.ply"))

    # 2 创建输出文件地址
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # 3 批量处理
    sample_grid = 0.4  # 采样大小：注意此处写死
    # 3.1 使用for循环方式处理
    for org_ply in sceneList:
        preprocess_BimTwins(org_ply, config.output_root, config.parse_normals, sample_grid, class_list=class_list,
        new_class=new_class)
    # 3.2 使用进程池方式处理（很慢，需要排查）
    # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())  # 创建进程池，所有cpu并行，构建一个
    # # pool = ProcessPoolExecutor(max_workers=6)
    # _ = list(
    #     pool.map(    # pool.map(任务名，参数1列表,...,参数n列表)；此处多进程异步执行函数handle_process
    #         preprocess_BimTwins,
    #         sceneList,
    #         repeat(config.output_root),     # itertools.repeat()
    #         repeat(config.parse_normals),
    #         repeat(sample_grid),
    #     )
    # )


    # 【test】 测试用代码
    # 预处理后文件
    # content = torch.load('/home/xuek/桌面/Pointcept/data/scannet/train/scene0000_00.pth')
    # content2 = torch.load('/home/xuek/桌面/Pointcept/data/scannet/train/scene0000_00.pth')

    # 【后处理】 postprocess【打标签】
    # labelPath ='/home/xuek/桌面/Pointcept/exp/bimtwins/semseg-pt-v3m1-0-base/result/JXY1_21_pred.npy'
    # dataPath = '/home/xuek/桌面/Pointcept/data_pre/btdata/train/JXY1_21.ply'
    # outputPath = '/home/xuek/桌面/Pointcept/data_pre'
    # postProcess(dataPath, labelPath, outputPath)  # 后处理代码
    # handle_process_BimTwins(scene_paths[0], config.output_root, train_scenes, val_scenes, config.parse_normals)

    # 【预处理】
    # preprocess_BimTwins(config.dataset_root, config.output_root, config.parse_normals)
    # # # 3 Preprocess data.-并行计算
    # # print("Processing scenes...")
    # # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())  # 创建进程池，所有cpu并行，构建一个
    # # # pool = ProcessPoolExecutor(max_workers=1)
    # #
    # # # 4 bt数据读入
    # # _ = list(
    # #     pool.map(    # pool.map(任务名，参数1列表,...,参数n列表)；此处多进程异步执行函数handle_process
    # #         preprocess_BimTwins,
    # #         sceneList,
    # #         repeat(config.output_root),     # itertools.repeat()
    # #         repeat(config.parse_normals),
    # #     )
    # # )