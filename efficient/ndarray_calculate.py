import numpy as np

def create_index_array(old_indices, new_indices):
    """
    建立两个ndarray的映射关系
        生成一个1D NumPy数组，将旧面索引映射到新面索引。
        用法：sublist_new = new_old_idx[sublist_old]
    注意：
        1）两个list元素一一对应，（不存在重复？）

    Args:
        old_indices: ndarray(n, 3), 旧面索引。
        new_indices: ndarray(n, 3), 新面索引。
    Returns:
        np.ndarray: 形状为 (n,) 的1D数组，每个元素是对应旧索引的新索引位置。
                    无法映射的索引返回 -1。
                    如果输入不一致，返回 None。
    """
    # 验证输入
    if old_indices.shape != new_indices.shape or old_indices.ndim != 2:
        return None

    n = old_indices.shape[0]

    # 1. 将每行转换为结构化类型（内存高效）
    dtype = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4')])
    old_struct = old_indices.copy().view(dtype).reshape(n)
    new_struct = new_indices.copy().view(dtype).reshape(n)

    # 建立新旧映射
    # 计算新旧结构到标准排序的映射
    old_to_std = np.argsort(old_struct)  # 旧结构 -> 标准排序的索引
    inv_idx1 = np.argsort(old_to_std)  # 上面是old_struct排序索引，而inv_idx1是作用在label索引的索引。

    new_to_std = np.argsort(new_struct)  # 新结构 -> 标准排序的索引
    # array_ordered = old_struct[old_to_std]  # 标准排序
    # 计算标准排序到新结构的逆向映射（即新结构的argsort的逆）
    std_to_new = np.argsort(new_to_std)
    # 组合映射：旧结构 -> 标准 -> 新结构
    old_to_new = new_to_std[inv_idx1]  # 注意顺序

    # 验证映射的正确性
    # a= np.array([1,2,3,4,5,6,7])
    # d= old_to_new[a]
    # ooo = old_indices[a]
    # nnn = new_indices[d]

    return old_to_new


def dict_common(data1,data2):
    '''
    获取两个数据的交集，快速求公共部分
    '''
    data1 = set(data1)
    data2 = set(data2)
    return_dict = data1 & data2
    return return_dict