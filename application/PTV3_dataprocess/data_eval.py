'''
评估训练数据质量
'''
import copy
import os
import sys
import torch
import json
import numpy as np
import path.path_process as path_process
import points.points_eval as points_eval
import data_dict

# 将字符串控制台输出并写出到文件
def PW(text, file_path):
    print(text)
    with open(file_path, 'a') as f:
        f.write(text + '\n')

# 1 评估训练数据质量
def eval_train_data(data_folder, output_json=None):  #, data_dict, label_version
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
    print("当前对点云分割数据进行评估，总共标注类别有5类")
    print("当前处理的文件为 pth 数据")

    # 初始化数据结构，按照JSON格式
    result = {
        "元数据": {
            "生成时间": "",
            "数据说明": {
                "总体说明": "本文件统计了点云分割数据集的分布信息，包括全局统计、各数据集统计以及场景级详细信息。",
                "层级说明": {
                    "全局级别": "统计所有数据集的汇总信息",
                    "数据集级别": "分别统计train/val/test三个数据集的信息",
                    "场景级别": "统计每个场景的详细信息"
                },
                "指标含义": {
                    "场景数量": "包含的场景文件总数",
                    "点云数量": "该层级包含的所有点云总数",
                    "类别点云数量": "某个类别（如背景、建筑）的点云数量",
                    "占总点云比例": "类别点云数量 / 全局总点云数量",
                    "占数据集比例": "类别点云数量 / 该数据集总点云数量",
                    "占场景比例": "类别点云数量 / 该场景总点云数量",
                    "子类分布": {
                        "说明": "某个类别中子类别的分布情况",
                        "计算规则": {
                            "场景级别": "存在的子类别平均分配（例如：同时存在平坦和斜坡，则各占50%）",
                            "数据集级别": "子类实际数量占比（子类点云数 / 该类别所有子类点云总数）",
                            "全局级别": "子类实际数量占比（子类点云数 / 该类别所有子类点云总数）"
                        }
                    }
                },
                "类别定义": {
                    "背景类": "包含平坦、斜坡等地面类型",
                    "建筑类": "包含高楼、民房、临建等建筑类型",
                    "车辆类": "包含普通汽车、工程汽车等车辆类型",
                    "高植被类": "树木等高大植被",
                    "低植被类": "草地、灌木等低矮植被"
                }
            }
        },
        "全局统计": {
            "场景数量": 0,
            "总点云数量": 0,
            "各类别统计": {}
        },
        "数据集分布": {}
    }

    # 类别名称映射
    label_names = {
        0: "背景类",
        1: "建筑类",
        2: "车辆类",
        3: "高植被类",
        4: "低植被类"
    }

    # 子类别名称映射
    sublabel_names = {
        "平坦": "平坦",
        "斜坡": "斜坡",
        "高楼": "高楼",
        "民房": "民房",
        "临建": "临建",
        "普通汽车": "普通汽车",
        "工程汽车": "工程汽车"
    }

    # 1 点云信息
    sub_folders = ['train', 'val', 'test']
    global_total_points = 0
    global_total_scenes = 0
    global_category_info = {}

    # 初始化全局类别统计
    for label_id in label_names:
        global_category_info[label_names[label_id]] = {
            "类别点云数量": 0,
            "占总点云比例": 0.0,
            "子类分布": {}
        }
        # 添加子类别字段
        if label_id == 0:  # 背景
            global_category_info[label_names[label_id]]["子类分布"]["平坦"] = 0.0
            global_category_info[label_names[label_id]]["子类分布"]["斜坡"] = 0.0
        elif label_id == 1:  # 建筑
            global_category_info[label_names[label_id]]["子类分布"]["高楼"] = 0.0
            global_category_info[label_names[label_id]]["子类分布"]["民房"] = 0.0
            global_category_info[label_names[label_id]]["子类分布"]["临建"] = 0.0
        elif label_id == 2:  # 车辆
            global_category_info[label_names[label_id]]["子类分布"]["普通汽车"] = 0.0
            global_category_info[label_names[label_id]]["子类分布"]["工程汽车"] = 0.0

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(data_folder, sub_folder)
        if not os.path.exists(sub_folder_path):
            continue

        # 初始化数据集信息
        result["数据集分布"][sub_folder] = {
            "数据集统计": {
                "场景数量": 0,
                "数据集总点云": 0,
                "各类别统计": copy.deepcopy(global_category_info)
            },
            "场景详情": {}
        }

        dataset_total_points = 0
        dataset_scene_count = 0
        dataset_category_info = {}

        # 初始化数据集类别统计
        for label_id in label_names:
            dataset_category_info[label_names[label_id]] = {
                "类别点云数量": 0,
                "占数据集比例": 0.0,
                "子类分布": {}
            }
            # 添加子类别字段
            if label_id == 0:  # 背景
                dataset_category_info[label_names[label_id]]["子类分布"]["平坦"] = 0.0
                dataset_category_info[label_names[label_id]]["子类分布"]["斜坡"] = 0.0
            elif label_id == 1:  # 建筑
                dataset_category_info[label_names[label_id]]["子类分布"]["高楼"] = 0.0
                dataset_category_info[label_names[label_id]]["子类分布"]["民房"] = 0.0
                dataset_category_info[label_names[label_id]]["子类分布"]["临建"] = 0.0
            elif label_id == 2:  # 车辆
                dataset_category_info[label_names[label_id]]["子类分布"]["普通汽车"] = 0.0
                dataset_category_info[label_names[label_id]]["子类分布"]["工程汽车"] = 0.0

        filelist = path_process.get_files_by_format(sub_folder_path, formats=['.pth'], return_full_path=False)

        for file in filelist:
            scenei_info = dict()
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(sub_folder_path, file)
            print(f"处理文件: {file_name}")
            pth_info = torch.load(file_path)
            label = pth_info['semantic_gt20']
            scenei_info['pts_num'] = label.size
            scenei_info['labels'] = points_eval.label_rate(label)

            # 计算子类别占比
            scenei_info_with_sublabels = calculate_sublabel_ratio(scenei_info, file_name)

            # 更新数据集统计
            dataset_scene_count += 1
            dataset_total_points += scenei_info['pts_num']

            # 场景信息
            scene_category_info = {}

            for label_id, label_info in scenei_info_with_sublabels['labels'].items():
                category_name = label_names.get(label_id, f"类别{label_id}")

                # 更新数据集类别统计
                dataset_category_info[category_name]["类别点云数量"] += label_info['数量']

                # 更新场景类别信息
                scene_category_info[category_name] = {
                    "类别点云数量": label_info['数量'],
                    "占场景比例": label_info['比例'],
                    "子类分布": {}
                }

                # 添加子类别信息
                # 收集该类别的所有子类别
                subcategories = []
                for key in label_info:
                    if key not in ['数量', '比例']:
                        subcategories.append(key)

                # 计算子类别在父类中的占比（平均分配）
                if len(subcategories) > 0:
                    sub_ratio = 1.0 / len(subcategories)
                    for sub in subcategories:
                        # 场景级别：子类别占父类的比例
                        scene_category_info[category_name]["子类分布"][sub] = sub_ratio

                        # 数据集级别：累计子类别数量
                        if sub in dataset_category_info[category_name]["子类分布"]:
                            dataset_category_info[category_name]["子类分布"][sub] += label_info[sub]

            # 添加场景信息到结果
            result["数据集分布"][sub_folder]["场景详情"][file_name] = scene_category_info

            # 输出子类别信息
            print(f"  子类别分布信息:")
            for label_id, label_info in scenei_info_with_sublabels['labels'].items():
                info_str = f"    标签{label_id}: 数量={label_info['数量']}, 比例={label_info['比例']:.2%}"
                # 添加子类别信息
                for key, value in label_info.items():
                    if key not in ['数量', '比例']:
                        info_str += f", {key}={value:.0f}"
                print(info_str)

        # 更新数据集信息
        result["数据集分布"][sub_folder]["数据集统计"]["场景数量"] = dataset_scene_count
        result["数据集分布"][sub_folder]["数据集统计"]["数据集总点云"] = dataset_total_points

        # 计算数据集类别占比
        for category_name in dataset_category_info:
            if dataset_total_points > 0:
                dataset_category_info[category_name]["占数据集比例"] = dataset_category_info[category_name]["类别点云数量"] / dataset_total_points

            # 计算子类别占比（子类别在父类中的占比）
            parent_count = dataset_category_info[category_name]["类别点云数量"]
            if parent_count > 0 and "子类分布" in dataset_category_info[category_name]:
                # 首先统计所有子类别的总数
                total_sub_count = 0
                for sub in dataset_category_info[category_name]["子类分布"]:
                    total_sub_count += dataset_category_info[category_name]["子类分布"][sub]

                # 然后计算每个子类别在父类中的占比（总和为1）
                if total_sub_count > 0:
                    for sub in dataset_category_info[category_name]["子类分布"]:
                        dataset_category_info[category_name]["子类分布"][sub] = dataset_category_info[category_name]["子类分布"][sub] / total_sub_count

        result["数据集分布"][sub_folder]["数据集统计"]["各类别统计"] = dataset_category_info

        # 更新全局统计
        global_total_scenes += dataset_scene_count
        global_total_points += dataset_total_points

        for category_name in global_category_info:
            global_category_info[category_name]["类别点云数量"] += dataset_category_info[category_name]["类别点云数量"]

            # 累计子类别数量
            if "子类分布" in global_category_info[category_name] and "子类分布" in dataset_category_info[category_name]:
                for sub in global_category_info[category_name]["子类分布"]:
                    if sub in dataset_category_info[category_name]["子类分布"]:
                        global_category_info[category_name]["子类分布"][sub] += dataset_category_info[category_name]["子类分布"][sub]

    # 添加生成时间到元数据
    import datetime
    result["元数据"]["生成时间"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 更新全局信息
    result["全局统计"]["场景数量"] = global_total_scenes
    result["全局统计"]["总点云数量"] = global_total_points

    # 计算全局类别占比
    for category_name in global_category_info:
        if global_total_points > 0:
            global_category_info[category_name]["占总点云比例"] = global_category_info[category_name]["类别点云数量"] / global_total_points

        # 计算子类别占比（子类别在父类中的占比）
        parent_count = global_category_info[category_name]["类别点云数量"]
        if parent_count > 0 and "子类分布" in global_category_info[category_name]:
            # 首先统计所有子类别的总数
            total_sub_count = 0
            for sub in global_category_info[category_name]["子类分布"]:
                total_sub_count += global_category_info[category_name]["子类分布"][sub]

            # 然后计算每个子类别在父类中的占比（总和为1）
            if total_sub_count > 0:
                for sub in global_category_info[category_name]["子类分布"]:
                    global_category_info[category_name]["子类分布"][sub] = global_category_info[category_name]["子类分布"][sub] / total_sub_count

    result["全局统计"]["各类别统计"] = global_category_info

    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    # 输出JSON结果
    if output_json:
        # 转换numpy类型
        result_converted = convert_numpy(result)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result_converted, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到: {output_json}")

    return result



    # 2 输出
    PW("数据质量评估结果：", eval_file)
    # 2.1 输出算量
    PW("总数据集评估：", eval_file)







        # 2 评估推理结果问题
# 混淆矩阵



# 3 评估模型问题


# 新增函数：计算子类别占比
def calculate_sublabel_ratio(scene_info, scene_name):
    '''
    计算场景中各个类的子类别占比

    参数:
        scene_info: 包含标签信息的字典，格式为 {'pts_num': int, 'labels': dict}
                    其中 labels 是标签分布字典，如 {0: {'数量': c, '比例': p}, 1: {...}, ...}
        scene_name: 场景名称，用于从 label_rate 中获取对应的子类别

    返回:
        修改后的 scene_info，其中 labels 字典增加了子类别信息
        格式: label: {'数量': c, '比例': p, '平坦': c*0.5, '斜坡': c*0.5, ...}
    '''
    return update_scene_info_with_sublabels(scene_info, scene_name)


def update_scene_info_with_sublabels(scene_info, scene_name):
    '''
    根据场景名称更新 scene_info，添加子类别占比信息

    参数:
        scene_info: 包含标签信息的字典，格式为 {'pts_num': int, 'labels': dict}
        scene_name: 场景名称，用于从 label_rate 中获取对应的子类别

    返回:
        修改后的 scene_info，其中 labels 字典增加了子类别信息
    '''
    # 复制输入字典以避免修改原始数据
    scene_info_modified = copy.deepcopy(scene_info)

    # 获取该场景的子类别列表
    if scene_name in data_dict.label_rate:
        scene_sublabels = data_dict.label_rate[scene_name]

        # 遍历每个标签类别
        for label_id, label_info in scene_info_modified['labels'].items():
            count = label_info['数量']
            ratio = label_info['比例']

            # 根据标签ID确定对应的子类别
            label_subcategories = []
            if label_id == 0:  # 背景
                label_subcategories = [sub for sub in scene_sublabels if sub in data_dict.sub_label["背景"]]
            elif label_id == 1:  # 建筑
                label_subcategories = [sub for sub in scene_sublabels if sub in data_dict.sub_label["建筑"]]
            elif label_id == 2:  # 车辆
                label_subcategories = [sub for sub in scene_sublabels if sub in data_dict.sub_label["车辆"]]

            # 计算子类别占比
            if len(label_subcategories) > 0:
                sub_ratio = 1.0 / len(label_subcategories)
                for subcategory in label_subcategories:
                    scene_info_modified['labels'][label_id][subcategory] = count * sub_ratio

    return scene_info_modified


if __name__ == '__main__':
    data_folder = r'/media/xuek/Data210/数据集/训练集/重建数据_动态维护_pth'
    output_json = r'/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/数据分布信息.json'

    eval_train_data(data_folder, output_json)