import pandas as pd

def generate_scene_group_csv(output_csv=None, data_version=None):
    '''
    根据data_dict.py中的data_version20240905创建场景分组CSV表格

    参数:
        output_csv: 输出CSV文件路径（可选）
        data_version: 数据版本字典（可选，默认使用data_version20240905）

    返回:
        DataFrame包含分组和场景名信息
    '''
    # 导入data_dict
    from application.PTV3_dataprocess import data_dict

    # 如果没有指定data_version，使用默认版本
    if data_version is None:
        data_version = data_dict.data_version20240905

    print(f"\n生成场景分组CSV表格...")
    print(f"数据版本: data_version20240905")

    # 存储所有数据
    all_data = []

    # 遍历每个分组
    for group_name, scene_set in data_version.items():
        # 遍历该分组中的每个场景
        for scene_name in scene_set:
            all_data.append({
                '分组': group_name,
                '场景名': scene_name
            })

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    # 按分组排序，然后按场景名排序
    if len(df) > 0:
        # 按分组顺序排序（train -> val -> test）
        group_order = {'train': 0, 'val': 1, 'test': 2}
        df['_group_order'] = df['分组'].map(group_order)
        df = df.sort_values(['_group_order', '场景名'])
        df = df.drop('_group_order', axis=1)
        df = df.reset_index(drop=True)

    # 保存到CSV
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
    output_csv =r'/home/xuek/桌面/PTV3_Versions/Common_3D/application/PTV3_dataprocess/输出测试.csv'
    generate_scene_group_csv(output_csv)