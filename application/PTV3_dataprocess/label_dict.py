'''
    该文档记录各版本标注数据类别，以及类别映射关系。
'''

label12_V1 = {
    0: "background",    1: "building",  2: "wigwam",    3: "car",
    4: "vegetation",    5: "farmland",  6: "shed",      7: "stockpiles",
    8: "bridge",        9: "pole",      10: "others",   11: "grass"
}

label06_V1 = {
    0: "background",   1: "building",     2: "car",     3: "vegetation",
    4: "farmland",     5: "grass"
}
label05_V1 = {
    0: "background",   1: "building",     2: "car",     3: "vegetation",
    4: "grass"
}

label04_V1 = {
    0: "background",   1: "building",     2: "vegetation",     3: "farmland"
}


# 创建标签映射
label_map_l12v1_to_l06v1 = {0: [0, 7, 9, 10],
                             1: [1, 2, 6, 8],
                             2: [3, ],
                             3: [4, ],
                             4: [5, ],
                             5: [11, ]}

label_map_l12v1_to_l05v1 = {0: [0, 7, 9, 10],
                            1: [1, 2, 6, 8],
                            2: [3, ],
                            3: [4, 5],
                            4: [11, ]}

label_map_l12v1_to_l04v1 = {0: [0, 7, 9, 10, 3, 11],
                            1: [1, 2, 6, 8],
                            2: [4, ],
                            3: [5, ]}
