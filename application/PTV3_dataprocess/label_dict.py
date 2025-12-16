'''
    该文档记录各版本标注数据类别，以及类别映射关系。
'''
class LabelRegistry:
    """Centralized registry for label definitions and mappings"""
    # Label definitions
    label_def = {
        'label12_V1': {
            0: "background", 1: "building", 2: "wigwam", 3: "car",
            4: "vegetation", 5: "farmland", 6: "shed", 7: "stockpiles",
            8: "bridge", 9: "pole", 10: "others", 11: "grass"
        },
        'label06_V1': {
            0: "background", 1: "building", 2: "car", 3: "vegetation",
            4: "farmland", 5: "grass"
        },
        'label05_V1': {
            0: "background", 1: "building", 2: "car", 3: "vegetation",
            4: "grass"
        },
        'label04_V1': {
            0: "background", 1: "building", 2: "vegetation", 3: "farmland"
        }
    }

    # Label mappings between different versions
    label_mappings = {
        'label12_V1_to_label06_V1': {
            0: [0, 7, 9, 10],
            1: [1, 2, 6, 8],
            2: [3],
            3: [4],
            4: [5],
            5: [11]
        },
        'label12_V1_to_label05_V1': {
            0: [0, 7, 9, 10],
            1: [1, 2, 6, 8],
            2: [3],
            3: [4, 5],
            4: [11]
        },
        'label12_V1_to_label04_V1': {
            0: [0, 7, 9, 10, 3, 11],
            1: [1, 2, 6, 8],
            2: [4],
            3: [5]
        },
        'label04_V1_to_label05_V1': {
            0: [0],
            1: [1],
            2: [3],
            3: [2]
        }
    }

    @staticmethod
    def get_labels_property_name(property_names):
        '''
        获取可能的标签名，即包含"class、label"的变量，且不包含坐标等名称。
        '''
        lable_names = set([property_name for property_name in property_names if 'label' in property_name
                       or 'class' in property_name])
        other_property_names = set([property_name for property_name in property_names if 'color' in property_name
                       or 'coord' in property_name])
        return lable_names - other_property_names

    @staticmethod
    def labels_correct_from_ply(labels_in):
        '''
        筛选所有标签名，并更正。
        专门为ply文件读写使用的标签矫正，去除ply文件读入后的前缀element名，如"vertex_", "label_"
        注意：
            1、当前只处理"vertex_", "label_"开头的标签名，不删除coords等已输入的标签。
            2、当前对于同名标签会重复放入，未增加筛选机制。
        参数：
            labels_in   输入标签，“elenmet名_label名”，如"vertex_class", "label_label12_V1"
            return      去除前缀"vertex_", "label_"的结果
        '''
        corrected_labels = []
        for label in labels_in:
            # 去除前缀，如"label_" 或 "vertex_"
            if label.startswith('vertex_'):
                label = label[7:]
            elif label.startswith('label_'):
                label = label[6:]
            corrected_labels.append(label)
        if len(corrected_labels) == 0:
            raise Exception("No 'pcl_elemnet style' label found!")
        return corrected_labels

    @staticmethod
    def label_element_style(lable_name):
        '''
        label转 ply的 element风格，如{'vertex_label05_V1','label_label12_V1'}
        '''
        return set('vertex_' + lable_name, 'label_' + lable_name)

    @classmethod
    def label_map(cls, labels_in_ply, label_out):
        '''
        筛选最优标签、及对应的映射关系。
        注意：
            1、兼容ply的element格式的类别命名，如{'label_label05_V1','label_label12_V1'}
        参数：
            labels_in_ply：标签列表，如{'label05_V1','label12_V1'}  （注意：去除ply element格式的标签）
            label_out：目标标签，如'label05_V1'
        return
            label_name  选择的最佳标签名
            mapping     最佳映射关系
        '''
        labels_in = cls.labels_correct_from_ply(labels_in_ply)  # 去掉前缀的标签
        if labels_in.size == 0:
            raise Exception("labels_in is empty!")

        # 1 同标签不修改
        if label_out in labels_in:
            return label_out, dict()  # 返回原标签
        # 2 最优版本标签
        available_label_name = set()
        if 'label12_V1' in labels_in:
            available_label_name.add('label12_V1')
        else:
            available_label_name = set(cls.label_def.keys()) & set(labels_in)  # 可用标签
            if len(available_label_name) == 0:
                raise Exception("No available label!")
        # 3 计算标签映射
        for label_name in available_label_name:
            mapping_name = label_name + '_to_' + label_out
            if mapping_name not in cls.label_mappings:
                continue
            else:  # label_name 映射回原始列表中
                mapping = cls.label_mappings[mapping_name]
                if label_name in labels_in_ply:
                    return_name = label_name
                else:
                    ply_lable_name_set = cls.label_element_style(label_name)
                    ply_lable_name_set.add(label_name)
                    return_name = ply_lable_name_set.intersection(labels_in_ply).pop()
                return return_name, mapping
        raise Exception("No mapping found!")

    @classmethod
    def label_map(cls, labels_in, label_out):
        '''
        筛选最优标签、及对应的映射关系。
        参数：
            labels_in：标签列表，如{'label05_V1','label12_V1'}  （注意：去除ply element格式的标签）
            label_out：目标标签，如'label05_V1'
        return
            label_name  选择的最佳标签名
            mapping     最佳映射关系
        '''
        if len(labels_in) == 0:
            raise Exception("labels_in is empty!")

        # 1 同标签不修改
        if label_out in labels_in:
            return label_out, dict()  # 返回原标签
        # 2 最优版本标签
        available_label_name = set()
        if 'label12_V1' in labels_in:
            available_label_name.add('label12_V1')
        else:
            available_label_name = set(cls.label_def.keys()) & set(labels_in)  # 可用标签
            if len(available_label_name) == 0:
                raise Exception("No available label!")
        # 3 计算标签映射
        for label_name in available_label_name:
            mapping_name = label_name + '_to_' + label_out
            if mapping_name not in cls.label_mappings:
                continue
            else:
                mapping = cls.label_mappings[mapping_name]
                return label_name, mapping
        raise Exception("No mapping found!")







