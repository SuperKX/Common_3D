'''
    本文件只用于记录各个版本的数据分配
'''
import pandas as pd


data_version20240905 = {
    'train': {'01JXY12', '02YTXC', '04HXKJC', '05YLZY1',
              '06XCNC', '07DJXZ1', '09Henan8', '12JKQ17_2',
              '24JKQ17_1', '25JKQ18', '26GZGD', '27DATA19_train',
              '29XY8B', '32_train', '33ZJ', '34PTY1',
              '35SBL_train', '36LT', '37XCNN_train', 'terrain3',
              'terrain9', 'UrbanBIS_Yingrenshi'},
    'val': {'01JXY3', '05YLZY2', '07DJXZ2', '27DATA19_val',
            '32_val', '35SBL_val', '37XCNN_val', '38HZ',
            '39PTY2', 'terrain16', 'terrain18'},
    'test': {}
}


sub_label={
    "背景": ["平坦", "斜坡"],
    "建筑": ["高楼", "民房", "临建"],
    "车辆": ["普通汽车", "工程汽车"],  # 修正为"工程汽车"以保持一致
}

label_rate = dict()
# train 数据集
label_rate['01JXY12'] = ["平坦","高楼","普通汽车"]
label_rate['02YTXC'] = ["平坦","高楼","普通汽车"]
label_rate['04HXKJC'] = ["平坦","高楼","普通汽车"]
label_rate['05YLZY1'] = ["平坦","高楼","普通汽车"]
label_rate['06XCNC'] = ["平坦","民房","普通汽车"]
label_rate['07DJXZ1'] = ["斜坡","民房","普通汽车"]
label_rate['09Henan8'] = ["平坦","民房","工程汽车"]
label_rate['12JKQ17_2'] = ["斜坡","平坦","民房","普通汽车"]
label_rate['24JKQ17_1'] = ["斜坡","平坦","民房","普通汽车"]
label_rate['25JKQ18'] = ["斜坡","平坦","民房","普通汽车"]
label_rate['26GZGD'] = ["平坦","民房","普通汽车"]
label_rate['27DATA19_train'] = ["斜坡","民房","普通汽车"]
label_rate['29XY8B'] = ["斜坡","民房","普通汽车"]
label_rate['32_train'] = ["斜坡","民房","工程汽车"]
label_rate['33ZJ'] = ["斜坡","平坦","高楼","民房","临建","普通汽车"]
label_rate['34PTY1'] = ["斜坡","高楼","民房","工程汽车"]
label_rate['35SBL_train'] = ["斜坡","民房","工程汽车"]
label_rate['36LT'] = ["平坦","高楼","民房","临建","工程汽车"]
label_rate['37XCNN_train'] = ["斜坡","民房","临建","工程汽车"]
label_rate['terrain3'] = ["平坦","民房","工程汽车"]
label_rate['terrain9'] = ["平坦","民房","临建","工程汽车"]
label_rate['UrbanBIS_Yingrenshi'] = ["平坦","高楼","普通汽车"]

# val 数据集
label_rate['01JXY3'] = ["平坦","高楼","普通汽车"]
label_rate['05YLZY2'] = ["平坦","高楼","普通汽车"]
label_rate['07DJXZ2'] = ["斜坡","民房","普通汽车"]
label_rate['27DATA19_val'] = ["斜坡","民房","普通汽车"]
label_rate['32_val'] = ["斜坡","民房","工程汽车"]
label_rate['35SBL_val'] = ["斜坡","民房","工程汽车"]
label_rate['37XCNN_val'] = ["斜坡","民房","临建","工程汽车"]
label_rate['38HZ'] = ["斜坡","平坦","高楼","临建","普通汽车"]
label_rate['39PTY2'] = ["斜坡","临建","工程汽车"]
label_rate['terrain16'] = ["平坦","民房","临建","工程汽车","普通汽车"]
label_rate['terrain18'] = ["平坦","临建","工程汽车"]

# test 数据集 (未提供具体标签信息，留空)




