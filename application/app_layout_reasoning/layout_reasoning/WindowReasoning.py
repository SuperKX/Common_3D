import copy

import numpy as np
import cv2
import os
import random
import json

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
'''
2025.03.05 版本备份
● 关闭墙面目标合并。GGP-97048
'''

def initLayout(inputJson, inputImagesFolder, tempPath='', testMode=False):
    '''
    brief 初始化json中布局结果
    :param inputJson: 读入的JSON字符串（非文件地址,可节省IO）
    :param inputImagesFolder: 读入的图片文件夹
    :param tempPath: 临时文件地址，为空则不输出中间文件
    :param testMode: 测试模式，开启则输出中间文件
    :return: json文件解析为墙面推理数据结构
    '''
    if tempPath == '' or testMode == False:
        testMode = False
    else:
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
    layoutInputAll = []  # 所有图片原始布局(按照门、窗区分)
    for key, val in inputJson.items():
        if key == 'status' or key == 'errorMsg' or key == 'time':
            continue
        imagePath = os.path.join(inputImagesFolder, key + '.jpg')
        if not os.path.exists(imagePath):
            imagePath = os.path.join(inputImagesFolder, key + '.png')
            if not os.path.exists(imagePath):
                print("找不到图片{}".format(key + '.jpg'))

        # 门窗布局解析
        layoutInput = {}  # 单张图片原始布局
        layoutInput["name"] = key
        layoutInput["path"] = imagePath
        layoutInput["window"] = []
        layoutInput['door'] = []
        layoutInput["windowIdx"] = []  # 用于记录窗户json中的编号,后续用于索引修改(因为door和window混在一起)
        idx = 0
        for valkey in val:
            if valkey['category'] == 'window':
                window = valkey["bbox"]
                layoutInput['window'].append(window)
                layoutInput["windowIdx"].append(idx)
            elif valkey['category'] == 'door':
                door = valkey["bbox"]
                layoutInput['door'].append(door)
            else:
                print('不处理的标签值{}'.format(valkey['category']))
            idx += 1

        if testMode:
            # if key != '0AJe6aXIV6Oq_61':
            #     continue
            # 绘制
            img = cv2.imread(imagePath)
            for rect in layoutInput['window']:
                rect = list(map(int, rect))
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 250, 0), 6)
            for rect in layoutInput['door']:
                rect = list(map(int, rect))
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (250, 0, 0), 6)
            cv2.imwrite(os.path.join(tempPath, layoutInput["name"] + '_0Recog.jpg'), img)
        layoutInputAll.append(layoutInput)  # 无效？
    return layoutInputAll

def iouCal(xa0,xa1,xb0,xb1):
    '''
    brief 计算线段交并比
    :param xa0:
    :param xa1:
    :param xb0:
    :param xb1:
    :return:
    '''
    len1 = xa1-xb0
    len2 = xb1-xa0
    if len1<=0 or len2<=0:
        return 0
    else:
        return min(len1/len2,len2/len1)

def perCentCal(xa0,xa1,xb0,xb1):
    '''
    brief 计算ab交集对线段b所占比例（用于判断包含关系）
    :param xa0:线段a起点
    :param xa1:线段a终点
    :param xb0:线段b起点
    :param xb1:线段b终点
    :return:占比的值
    '''
    len1 = xa1-xb0
    len2 = xb1-xa0
    if len1<=0 or len2<=0:  # 相离 -1
        return -1
    else:
        rate = min(len1,len2)/(xb1-xb0)
        # if rate>1:  # 如果rate》1说明leni已经长过xb本身，一定为包含
        #     rate = 1
        return rate  # rate>1 为被包含关系

def Confusion_Matrix_ColRelation(windows):
    '''
    计算列方向相关性的混淆矩阵。
    :param windows:
    :return:
    '''
    winNum = len(windows)  # 总窗户数量
    confusionMtx = np.zeros((winNum, winNum), dtype=float)  # 构造混淆矩阵
    for i in range(winNum):
        for j in range(winNum):
            confusionMtx[i][j] = perCentCal(windows[i][0], windows[i][2], windows[j][0], windows[j][2])
    return confusionMtx

def ConfMtx_size(windows):
    '''
    计算两个窗户相似性，返回相似比例的权重。
    :param windows:
    :return:
    '''
    winNum = len(windows)  # 总窗户数量
    confusionMtx = np.zeros((winNum, winNum), dtype=float)  # 构造混淆矩阵
    for i in range(winNum):
        for j in range(winNum):
            if i==j:
                confusionMtx[i][j] =1
            elif i>j:
                wini_h = windows[i][3] - windows[i][1]
                winj_h = windows[j][3] - windows[j][1]
                h_rate = wini_h / winj_h if winj_h > wini_h else winj_h / wini_h
                confusionMtx[i][j] = h_rate
            else:
                wini_w = windows[i][2] - windows[i][0]
                winj_w = windows[j][2] - windows[j][0]
                w_rate = wini_w / winj_w if winj_w > wini_w else winj_w / wini_w
                confusionMtx[i][j] = w_rate
            # confusionMtx[i][j] = h_rate*w_rate # 要求长宽均要满足比例需求
    return confusionMtx

def ConfMtx_pos(windows):
    '''
    计算两个窗户w和h方向的iou
    :param windows:
    :return:
    '''
    winNum = len(windows)  # 总窗户数量
    confusionMtx = np.zeros((winNum, winNum), dtype=float)  # 构造混淆矩阵
    for i in range(winNum):
        for j in range(winNum):
            if i==j:
                confusionMtx[i][j] = 1
            # 1) h方向iou
            elif i > j:  # i>j 记录h方向iou
                if windows[i][3] < windows[j][1] or windows[j][3] < windows[i][1]:
                    confusionMtx[i][j] = 0
                else:
                    iouh = (min(windows[i][3], windows[j][3]) - max(windows[i][1], windows[j][1])) / (max(windows[i][3], windows[j][3]) - min(windows[i][1], windows[j][1]))
                    confusionMtx[i][j] = iouh
            # 2 w方向iou
            elif i < j:  # i>j 记录w方向iou
                if windows[i][2] < windows[j][0] or windows[j][2] < windows[i][0]:
                    confusionMtx[i][j] = 0
                else:
                    iouw = (min(windows[i][2], windows[j][2]) - max(windows[i][0], windows[j][0])) / (max(windows[i][2], windows[j][2]) - min(windows[i][0], windows[j][0]))
                    confusionMtx[i][j] = iouw
    return confusionMtx

def connectGroup_ADD(boolMarix):
    '''
    深度优先搜索，区分联通区域
    :param boolMarix:
    :return:
    '''
    n=boolMarix.shape[0]
    visited=np.zeros([n,n],dtype=bool)  # 标记是否访问
    components=[]  # 所有联通分量列表

    def dfs(node, neighbor1, component):
        if not visited[node][neighbor1]:
            visited[node][neighbor1] = True
            visited[neighbor1][node] = True
            visited[neighbor1][neighbor1] = True
            component.append(neighbor1)
            for neighbor in range(n):
                if neighbor not in component:  # 查找值已入栈
                    # 判断条件 and
                    if (boolMarix[neighbor1, neighbor] and not visited[neighbor1][neighbor])\
                            and (boolMarix[neighbor, neighbor1] and not visited[neighbor][neighbor1]):
                        dfs(neighbor1,neighbor,component)

    for node in range(n):
        if not visited[node][node]:
            component=[]
            # visited[node][node]=True
            dfs(node,node,component)
            components.append(component)
    return components

def initColList(boolMarix):
    '''
    深度优先搜索，区分联通区域
    :param boolMarix:
    :return:
    '''
    n=boolMarix.shape[0]
    visited=np.zeros([n,n],dtype=bool)  # 标记是否访问
    components=[]  # 所有联通分量列表

    def dfs(node, neighbor1, component):
        if not visited[node][neighbor1]:
            visited[node][neighbor1] = True
            visited[neighbor1][node] = True
            visited[neighbor1][neighbor1] = True
            component.append(neighbor1)
            for neighbor in range(n):
                if neighbor not in component:  # 查找值已入栈
                    if (boolMarix[neighbor1, neighbor] and not visited[neighbor1][neighbor])\
                            or (boolMarix[neighbor, neighbor1] and not visited[neighbor][neighbor1]):
                        dfs(neighbor1,neighbor,component)

    for node in range(n):
        if not visited[node][node]:
            component=[]
            # visited[node][node]=True
            dfs(node,node,component)
            components.append(component)
    return components

def sameCol(listi,listj, thresh):
    '''
    列匹配
    :param listi:
    :param listj:
    :param thresh: 超过阈值表示合格
    :return:
    '''
    numMath_win =0  # 统计匹配窗户数量
    for wini in listi:
        for winj in listj:
            # 位置无交集则跳过
            if wini[3] < winj[1]:
                break
            if wini[1] > winj[3]:
                continue
            # 位置有交集计算高程交并比、及水平交并比。（大小位置统一为同窗）
            iou_height = (min(wini[3],winj[3])-max(wini[1],winj[1]))/(max(wini[3],winj[3])-min(wini[1],winj[1]))
            area_cross = min(wini[3]-wini[1], winj[3]-winj[1])*min(wini[2]-wini[0], winj[2]-winj[0])  # 面积交集
            area_add = (wini[3]-wini[1])*(wini[2]-wini[0])+(winj[3]-winj[1])*(winj[2]-winj[0])
            iou_weight = area_cross/(area_add-area_cross)  #(wini[3]-wini[1])/(winj[3]-winj[1])
            if iou_height>thresh and iou_weight>thresh:
                numMath_win+=1
    return numMath_win

def insertWin(winList, winSearch, windows):
    '''
    输入窗户列表，查找窗户所在列表中列位置，如果没有则返回插入位置。
    :param winList: 窗户id列表
    :param winSearch: 待查询窗户id
    :return:boolInsert, pos  #是否插入,位置
    '''
    boolInsert = False  # 默认不插入
    pos = -1
    # top, bot = -1, -1  # 查找时
    if len(winList)==0:  # 列表空的时候直接插入
        boolInsert = True
        pos = 0  # 插入当前位置
        return boolInsert, pos

    for i in range(len(winList)):
        winCompare = winList[i]  # 当前比较窗户的id
        winc = windows[winCompare]
        # winc1 = windows[winList[i+1]]  # 下一个窗
        wins = windows[winSearch]
        # 判断位置关系
        # 1) 是否插入当前位置
        if wins[3] < winc[1]:
            boolInsert = True
            pos = i  # 插入当前位置
            break
        iouh = iouCal(wins[1], wins[3], winc[1], winc[3])
        # 2) 是否当前同行
        if iouh > 0:  # 设置阈值
            # boolInsert = False  # 默认不插入
            pos = i
            break
        # 3) 跟下一个比较
        else:
            if i == len(winList) - 1:  # 最后一个
                boolInsert = True  # 插入末尾
                pos = i + 1
            else:
                continue
    return boolInsert, pos

def winStandard3(colsInfos, windows,stdWinsInfo): # 二维窗户list
    '''
    补充空窗
    :param colrelation: 列group
    :param colsInfos: 列信息
    :param windowsNP:
    stdWinsInfo 标准窗
    :return:
    '''
    # 返回值
    addWins=[]
    # 1 计算标准列：将所有标准窗排列 - 二维矩阵
    colrelation = colsInfos['col_group']  # 列组列表
    colWinIdxs = colsInfos['colWinIdxs']  # 窗户列表
    std_WinIdx = colsInfos['std_WinIdx']  # 标准窗索引

    for collist_k in colrelation:  # 逐个列组
        if len(collist_k)<2:  # 无标准列则排除
            continue
        # 0 列排序(后续构造顺序依赖列顺序)
        collist_k_w = np.array(colsInfos['std_center'])[collist_k]
        order = np.argsort(collist_k_w[:,0])
        collist_k=(np.array(collist_k)[order]).tolist()

        # 1.0 维护一个标准窗构成的列
        stdwinList = []  # 标准列: 二维列表,每行对应标准列的行,同行多个窗户则放入其中(需要!!:后面列匹配需要相邻关系)
        winMtx_mn = []  # 窗户矩阵: 记录矩阵中每个窗户编号,-1表示需要补充窗户
        # stdMtx_mn = []  # 标准行列矩阵,记录标准网格,每个窗对应的标准窗编号. -1表示非标准窗
        # 1.1 更新第一列数据
        collist_k0 = collist_k[0]  # 列编号
        stdwinList = [colWinIdxs[collist_k0][n_idx] for n_idx in
                      range(len(colWinIdxs[collist_k0])) if not std_WinIdx[collist_k0][n_idx] == -1]
        winMtx_0 = copy.deepcopy(stdwinList)
        winMtx_mn.append(winMtx_0)  # 注意:此处append是浅拷贝!!

        # 1.2 其他列补充
        for m in range(1, len(collist_k)):  # 逐个列
            collist_ki = collist_k[m]  # 列编号
            std_WinIdx_m = std_WinIdx[collist_ki]  # 当前列 标准窗
            colWinIdxs_m = colWinIdxs[collist_ki]  # 当前列 窗编号
            # 1) 初始化当前列信息
            winMtx_m = [-1] * len(stdwinList)  # 当前列的窗户
            for n in range(0, len(std_WinIdx_m)):  # 当前窗
                if std_WinIdx_m[n] == -1:  # 先不处理非标准窗
                    continue
                winSearch = colWinIdxs_m[n]
                winList = stdwinList
                # 2) 计算窗户关系
                boolInsert, pos = insertWin(winList, winSearch, windows)  # 当前窗对列排布的修改
                if pos == -1:
                    print('窗户位置匹配错误')
                # 3) 更新窗户关系
                if not boolInsert:  # 不插入值
                    stdwinList[pos] = winSearch  # 全局维护的标准列 [注意:使用最新列的窗来帮助近邻窗匹配问题]
                    winMtx_m[pos] = winSearch  # 当前列更新结果
                else:  # 如果要插入,前面所有列都插入
                    stdwinList.insert(pos, winSearch)  # 更新 标准列
                    winMtx_m.insert(pos, winSearch)  # 更新 当前列
                    for winMtx_before in winMtx_mn:  # 更新 之前列
                        winMtx_before.insert(pos, -1)
            # 4) 更新列关系
            winMtx_mn.append(winMtx_m)

        # 2 添加窗户
        # winMtx = np.array(winMtx_mn)  # np方便计算
        for i in range(len(winMtx_mn)):  # 遍历 标准列矩阵
            if not -1 in winMtx_mn[i]:  # 如果标准矩阵 当前列无需要补充的窗,则跳过
                continue
            # 2.1 非标准窗列表
            collist_ki = collist_k[i]  # 列编号
            std_WinIdx_m = std_WinIdx[collist_ki]  # 当前列 标准窗
            colWinIdxs_m = colWinIdxs[collist_ki]  # 当前列 窗编号
            ustd_list = [colWinIdxs_m[id] for id in range(len(std_WinIdx_m)) if std_WinIdx_m[id] == -1]  # 非标准窗列表

            # 顶部窗户不推理，防止多余窗户推到天上，2025.03.21
            bool_winsky = False  # 改为Ture表示继续推理天上窗户
            startj = 0
            if not bool_winsky:
                startj= next((j for j, x in enumerate(winMtx_mn[i]) if x != -1), 0)
            for j in range(startj, len(winMtx_mn[i])):  # 当前标准列逐个窗
                if winMtx_mn[i][j] == -1:
                    # 2.2 构造标准窗
                    # 1) 搜索最近窗户:同行的相邻窗户作为标准窗模板
                    neighbor = 1
                    newWin = []
                    while (i-neighbor >= 0) or (i+neighbor < len(winMtx_mn)):  # 水平方向左右找最近邻窗户
                        nei_left, nei_right = -1, -1
                        if i-neighbor >=0:
                            nei_left = winMtx_mn[i-neighbor][j]
                        if i+neighbor < len(winMtx_mn):
                            nei_right = winMtx_mn[i+neighbor][j]
                        if nei_left == -1 and nei_right == -1:
                            neighbor += 1
                            continue
                        winneighbors=[]
                        if not nei_left == -1:
                            winneighbors.append(nei_left)
                        if not nei_right == -1:
                            winneighbors.append(nei_right)
                        # 标准窗构建
                        neighbor_num = len(winneighbors)
                        centerw = colsInfos['std_center'][collist_ki][0]  # 中心坐标
                        sizeWH = colsInfos['std_size'][collist_ki]
                        newWin0 = centerw - sizeWH[0] / 2
                        newWin2 = centerw + sizeWH[0] / 2
                        newWin1, newWin3 = 0, 0
                        for win in winneighbors:
                            winNei = windows[win]
                            newWin1 += winNei[1] * 1 / neighbor_num
                            newWin3 += winNei[3] * 1 / neighbor_num
                        # 新增窗户
                        newWin = [newWin0, newWin1, newWin2, newWin3]
                        break
                    if newWin == []:
                        print("窗户计算错误")
                    # 搜素窗
                    # winSearch = newWin
                    # winList = ustd_list
                    # 2.3 计算窗户关系
                    addBool = True  # 添加窗户
                    for winC_idx in ustd_list:
                        winc = windows[winC_idx]
                        wins = newWin
                        iouh = iouCal(wins[1], wins[3], winc[1], winc[3])
                        if iouh > 1e-6:  # 判断:存在交集则不添加窗户
                            addBool = False
                            break
                    if addBool:  # 说明没有窗,需要插入
                        addWins.append(newWin)

    return addWins

def kmeansClaster(X, testMode,*args):
    '''
    k均值求聚类
    :param X:
    :return:
    '''
    # 使用K-means算法，假设划分为3个簇
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)  # 获取聚类标签

    if testMode:
        path,name,windows=args  # 解析
        # 1) plot
        # 绘制散点图
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')
        centers = kmeans.cluster_centers_  # 获取聚类中心
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50, alpha=0.75, marker='X')
        # 设置坐标轴范围从0开始
        plt.axis([0, max(X[:, 0])+100, 0, max(X[:, 1])+100])  # [xmin, xmax, ymin, ymax]
        plt.title(name)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        test_stop = 1
        plt.ioff()

        # 2) cv2
        img = cv2.imread(path)
        pic_h, pic_w, _ = img.shape
        color = []  # 不同聚类不同颜色
        for k in range(len(y_kmeans)):  # 构造颜色
            while len(color) < y_kmeans[k] + 1:
                colori=[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                while sum(colori)<255*3*0.5:  # 颜色过浅
                    colori=[(colori[0]+10)%255,(colori[1]+10)%255,(colori[2]+10)%255]
                color.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            rect = windows[k]
            rect = list(map(int, rect))
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (color[y_kmeans[k]]), int(pic_h / 250))
        # 压缩图片
        img_show_W=1000
        img_zip = cv2.resize(img, (img_show_W, int(img_show_W*pic_h/pic_w)), interpolation=cv2.INTER_AREA)
        cv2.imshow(name, img_zip)
        cv2.waitKey(0)  # 等待按键按下
        cv2.destroyAllWindows()  # 关闭所有窗口
        # cv2.imwrite(os.path.join(tempPath, wall["name"] + '_1colGroup.jpg'), img)
        test_here=1

    return y_kmeans

def colCorrect_DBL(stdWinsInfo,wincolIdx,windows):
    '''
    非单列,列矫正
    矫正列窗存在并排窗户的情况，并排窗户则合并并排窗，删除
    {更新窗户替换掉原始窗户、同时返回删除的编号}
    :param stdWinsInfo: 标准窗信息
    :param wincolIdx: 当前列窗户列表

    :return:
    '''
    # 返回值
    boolProcess = True  # False 表示该列为复杂的多列情况，不参与后续列匹配等工作。但是不代表该列窗户没有调整
    winIdx_new= {}  # 记录更新的窗户编号
    winIdx_del = set()  # 删除的窗户编号
    winIdx_refresh = set()  # 更新窗

    # newWinList=[]  # 新增窗户列表
    # wincolIdxNew = []  # 更新后窗户列表
    # winDeleteIdx = []  # 删除的窗户编号

    # 1 找到当前列的包含的标准窗的最大宽度 maxW
    stdlist = stdWinsInfo['list']
    stdWinSize = stdWinsInfo['size']
    standTrue=[]  # 记录当前列每个窗所属的标准窗类别
    maxW = -1  # 记录最大宽度
    for target in wincolIdx:  # 列中窗户编号
        indices = [i for i, row in enumerate(stdlist) for j, value in enumerate(row) if value == target]  # i返回所在的标准窗编号
        indice = indices[0] if len(indices)>0 else -1  # 当前列中窗户target对应的标准窗类别  #【待优化？】
        standTrue.append(indice)
        if indice>=0 and indice<len(stdWinSize):
            if stdWinSize[indice][0] > maxW:
                maxW = stdWinSize[indice][0]
    if maxW < 0:  # 没找到标准窗，不处理该列
        return False, winIdx_new, winIdx_del, winIdx_refresh  # 返回窗列表

    # 2 合并并列窗，计算是否可合并
    for i in range(len(wincolIdx) - 1):
        win = windows[wincolIdx[i]]
        win2 = windows[wincolIdx[i + 1]]
        if perCentCal(win[1], win[3], win2[1], win2[3]) > 0:  # 纵向相交
            mergeW = max(win[2],win2[2])-min(win[0],win2[0])  # 合并后窗户宽度
            # 1)合并后过宽，则不合并(暂不处理多列情况)
            if mergeW > maxW*1.1:
                # wincolIdxNew.append(wincolIdx[i])  # 窗户下标
                boolProcess = False  # 改种情况不处理
                continue
            # 2）合并后不很宽，则合并
            else:
                newWin = [min(win[0],win2[0]),min(win[1],win2[1]),max(win[2],win2[2]),max(win[3],win2[3])]
                winIdx_new[wincolIdx[i]] = newWin  # 记录更新的窗户编号
                winIdx_refresh.add(wincolIdx[i])
                winIdx_del.add(wincolIdx[i + 1])  # 删除的窗户编号

    return boolProcess, winIdx_new, winIdx_del, winIdx_refresh

def colCorrect_SIG(stdWinsInfo,wincolIdx,windows):

    '''
    矫正单列数据  {更新窗户替换掉原始窗户、同时返回删除的编号}
    1\解决窗被分成多个: 当前列中,紧靠窗户,如果合并后跟标准窗出现重合,则合并.
    :param stdWinsInfo: 标准窗信息
    :param wincolIdx: 当前列窗户列表
    :return:
    '''
    # 返回值
    boolProcess = True  # 是否进行后续处理, 单列目前都进行后续处理.
    winIdx_new= {}  # 记录更新的窗户编号
    winIdx_del = set()  # 删除的窗户编号
    winIdx_refresh = set()  # 更新窗

    # 需要信息
    # stdlist = stdWinsInfo['list']
    stdWinSize = stdWinsInfo['size']

    if len(wincolIdx)< 2:
        return boolProcess, winIdx_new, winIdx_del, winIdx_refresh
    for i in range(len(wincolIdx) - 1):
        # 1 查找前后仅靠窗
        win = windows[wincolIdx[i]]
        win2 = windows[wincolIdx[i + 1]]
        # 2 制造合并窗
        winH = win[3] - win[1]
        win2H = win2[3] - win2[1]
        dis = win2[1] - win[3]
        if dis < 0.3 * winH and dis < 0.3 * win2H:  # 判定条件: 两窗间隔过小判定可能误判识别
            newWin = [min(win[0], win2[0]), min(win[1], win2[1]), max(win[2], win2[2]), max(win[3], win2[3])]
            newWin_size = [newWin[2] - newWin[0], newWin[3] - newWin[1]]  # wh
            # 3 查看合并窗是否可以跟标准窗匹配
            for stdSize in stdWinSize:
                if not (newWin_size[0] > 0.9 * stdSize[0] and newWin_size[0] < 1.1 * stdSize[0]):
                    continue
                if newWin_size[1] > 0.9 * stdSize[1] and newWin_size[1] < 1.1 * stdSize[1]:
                    winIdx_new[wincolIdx[i]] = newWin  # 记录更新的窗户编号
                    winIdx_refresh.add(wincolIdx[i + 1])
                    winIdx_del.add(wincolIdx[i + 1])  # 删除的窗户编号
                    break
    return boolProcess, winIdx_new, winIdx_del, winIdx_refresh

def addWindow2(layoutInputAll,tempPath='',testMode=False):
    '''
    brief 补充缺失的窗户
    :param layoutInputAll: 初始布局-修改后的布局返回同一个
    :param tempPath: 临时文件夹
    :param testMode: 是否输出中间文件
    :return: layoutInputAll 同样返回
    '''
    if tempPath == '' or testMode == False:
        testMode = False
    else:
        tempPath = os.path.join(tempPath, "1_temp")
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
    for wall in layoutInputAll:  # 逐个墙面
        win_changed = []  # 替换后的窗户列表
        # 1 窗户信息补全
        # 1.1 根据列排序
        if testMode:
            print(wall['name'])
            if wall['name'] != '0AJe6aXIV6Oq_61':
                # print("测试目标数据")
                # continue
                stoptest = 1
        windows = wall['window']
        doors = wall['door']

        # 初始索引记录
        windowsNP = np.array(windows)
        idx_ORG = np.argsort(windowsNP[:,0])  # xuek-2025.02.22


        windows = sorted(windows, key=(lambda x: x[0]))
        # 1.2 计算窗户信息
        windowsNP = np.array(windows)
        # 信息矩阵[坐标[0-3]，w,h,中心点坐标]
        windows_WH=[[windows[i][2]-windows[i][0],windows[i][3]-windows[i][1]] for i in range(len(windows))]  # 窗户宽、高列表
        windows_Center=[[(windows[i][0]+windows[i][2])/2,(windows[i][1]+windows[i][3])/2] for i in range(len(windows))]  # 窗中心点
        # windowsNP=[windowsNP[i].extend([windowsNP[2]-windowsNP[0],windowsNP[3]-windowsNP[1],(windowsNP[0]+windowsNP[2])/2,(windowsNP[1]+windowsNP[3])/2]) for i in range(len(windowsNP))]
        # 1.3 计算几个混淆矩阵
        # 1）iou
        # 2）列位置
        # 3）行位置

        # 2 计算标准窗
        def standardWinds(sizeThresh):
            # 标准窗-尺寸混淆矩阵
            confMtx_size = ConfMtx_size(windows)  # 尺寸的混淆矩阵[i>j存储h比例，i<j存储w比例]
            confMtx_SizeBool = confMtx_size > sizeThresh  # 返回混淆矩阵合格矩阵
            # 深度优先搜索构建列关系
            StandWinIdxs = connectGroup_ADD(confMtx_SizeBool)  # 列索引分组,ex:[[1,2,3],[4,5]]
            StandWinIdxs = [StandWinIdx for StandWinIdx in StandWinIdxs if len(StandWinIdx) >= 3]  # 筛选 3个以上的联通
            # 标准窗计算尺寸
            stdWinSize = []
            for StandWinIdx in StandWinIdxs:
                a = np.array(windows_WH)[StandWinIdx]
                meanwh = np.mean(a, axis=0)
                stdWinSize.append([meanwh[0], meanwh[1]])
            stdWinsInfo = {}
            stdWinsInfo['list'] = StandWinIdxs
            stdWinsInfo['size'] = stdWinSize

            return stdWinsInfo


        # # 2.1 窗户聚类
        # X=np.array(windows_WH)
        # kmeans = kmeansClaster(X,True,wall['path'],wall['name'],windows)
        # # kmeans = kmeansClaster(X,False)

        # 2.2 列方向聚类
        # 1) 标准窗-尺寸混淆矩阵
        stdWinsInfo = standardWinds(0.9)

        # confMtx_size = ConfMtx_size(windows)  # 尺寸的混淆矩阵[i>j存储h比例，i<j存储w比例]
        # # 评价
        # sizeThresh = 0.9  # w、h都要满足
        # confMtx_SizeBool = confMtx_size > sizeThresh  # 返回混淆矩阵合格矩阵
        # # 深度优先搜索构建列关系
        # StandWinIdxs = connectGroup_ADD(confMtx_SizeBool)  # 列索引分组,ex:[[1,2,3],[4,5]]
        # StandWinIdxs = [StandWinIdx for StandWinIdx in StandWinIdxs if len(StandWinIdx)>=3]  # 筛选 3个以上的联通
        # # 标准窗计算尺寸
        # stdWinSize=[]
        # for StandWinIdx in StandWinIdxs:
        #     a = np.array(windows_WH)[StandWinIdx]
        #     meanwh = np.mean(a,axis=0)
        #     stdWinSize.append([meanwh[0],meanwh[1]])
        # stdWinsInfo={}
        # stdWinsInfo['list'] = StandWinIdxs
        # stdWinsInfo['size'] = stdWinSize

        # 2) 计算位置iou混淆矩阵
        confMtx_pos = ConfMtx_pos(windows)  # 尺寸的混淆矩阵[i>j存储h的iou，i<j存储w的iou]
        # 评价
        posThresh = 0.9  # w、h都要满足

        # 0 评价图片质量
        def pic_evaluate(confMtx_pos, stdWinsInfo):
            '''
            评价图片质量:通过水平方向存在交集的窗户iou均值,作为评价指标
            :param confMtx_pos:
            :return:
            '''
            wins_num, _ = confMtx_pos.shape
            relationW, relation_numW = 0, 0
            for i in range(wins_num):
                for j in range(wins_num):
                    if i < j and confMtx_pos[i][j] > 0:
                        relation_numW += 1
                        relationW += confMtx_pos[i][j]
            winstdnum = 0
            for wini in stdWinsInfo['list']:
                winstdnum += len(wini)
            # print(f'W方向: 平均iou= {relationW / (relation_numW + 1E-6)},  合格窗比例= {1.0 * winstdnum / (wins_num + 1E-6)}')
            return relationW / (relation_numW + 1E-6), 1.0 * winstdnum / (wins_num + 1E-6)

        iou_score, num_score = pic_evaluate(confMtx_pos, stdWinsInfo)
        if iou_score < 0.55:
            if num_score < 0.9:
                print(f"{wall['name']} quality {iou_score:.2f} at risk!")
                continue


        # a=np.array(windows)[:,0]
        # b=np.array(windows)[:,2]
        # c=np.vstack([a,b])
        # d=c.T
        # kmeans = kmeansClaster(d,True,wall['path'],wall['name'],windows)
        # # kmeans = kmeansClaster(X,False)

        # 3 布局分列
        # 3.1 计算列方向混淆矩阵
        confMtx_Col = Confusion_Matrix_ColRelation(windows)
        # 评价
        colThresh = 0.60  # 评价是否为同一列(超过改值判定为同列)
        confMtx_ColBool = confMtx_Col > colThresh  # 返回混淆矩阵合格矩阵
        # 深度优先搜索构建列关系
        windowColIdxs = initColList(confMtx_ColBool)  # 列索引分组,ex:[[1,2,3],[4,5]]

        # 4 列矫正          # 将多列细分为合格列
        colsInfos= {}  # 存储列信息的  # 'colWinIdxs'列数量；
        # colsInfos['colList']=[]
        colsInfos['colWinIdxs'] = []  # 记录每列的窗id  [[1,3,5],[7,9,2]]
        colsInfos['process'] =[]  # 记录是否当前列可以处理

        winIdx_news = {}  # 更新后窗
        winIdx_dels = set()  # 需要删除的窗
        winIdx_refreshs = set()  # 更改的窗户

        for wincolIdx in windowColIdxs:  # 逐列
            colsInfos['process'].append(True)  # 列可处理
            # 4.1 更新窗户顺序：每列按照高低顺序排序
            colInfo = {}  # 当前列信息
            wincol_temp = windowsNP[wincolIdx]  # 当前列窗户
            idx_temp = np.argsort(wincol_temp[:,1])  # 按照窗户起始高度排序后的序列索引
            wincolIdx = np.array(wincolIdx)[idx_temp]  # 更新后的窗户索引{总窗索引！} ndarray!
            wincol = windowsNP[wincolIdx]  # 更新后的窗户列表

            # 4.2 判断窗户是否单列
            bool_SingleCol = True
            for i in range(len(wincol)-1):
                win = wincol[i]
                win2 = wincol[i+1]
                if perCentCal(win[1], win[3], win2[1], win2[3])>0:
                    bool_SingleCol = False
                    break
            # 4.3 非单列进行处理  # 2025.03.05 暂时关闭，只关闭后续处理
            if not bool_SingleCol:
                colsInfos['process'][-1]=False
                # boolProcess, winIdx_new, winIdx_del, winIdx_refresh = colCorrect_DBL(stdWinsInfo, wincolIdx, windows)
                # colsInfos['process'][-1] = boolProcess  # 该列是否后续处理
                # winIdx_news = winIdx_news | winIdx_new
                # winIdx_dels.update(winIdx_del)
                # winIdx_refreshs.update(winIdx_refresh)
                # # 更新列信息（剔除列中错误窗）
                # wincolIdx = wincolIdx[~np.isin(wincolIdx, list(winIdx_del))]
            # 4.4 单列优化
            else:
                boolProcess, winIdx_new, winIdx_del, winIdx_refresh = colCorrect_SIG(stdWinsInfo, wincolIdx, windows)
                winIdx_news = winIdx_news | winIdx_new
                winIdx_dels.update(winIdx_del)
                winIdx_refreshs.update(winIdx_refresh)
                # 更新列信息（剔除列中错误窗）
                wincolIdx = wincolIdx[~np.isin(wincolIdx, list(winIdx_del))]
            colsInfos['colWinIdxs'].append(wincolIdx.tolist())  # 更新列信息


        # 4.4 窗户列表信息更新
        # 1）更新 windows
        for winID, win in winIdx_news.items():
            windows[winID] = win
            win_changed.append(win)
        # 1）更新 删除窗
        # for winID in winIdx_dels:
        #     windows[winID] = np.random.rand(4).tolist()  # 【待优化】如何处理删除窗
        # 更新 windowsNP
        windowsNP = np.array(windows)
        # 2）更新confMtx_pos
        confMtx_pos = ConfMtx_pos(windows)  #【！】待优化！！-只更新新的
        # 更新标准窗  #【！】待优化！！-只更新新的
        stdWinsInfo = standardWinds(0.9)

        # 4.5 计算列信息
        colsInfos['col_num'] = len(colsInfos['colWinIdxs'])  # 列数量
        colsInfos['std_size'] = []  # 列的标准窗形状 w h
        colsInfos['std_center'] = []  # 列中心坐标（x,y）
        colsInfos['std_dis'] = []  # 列间隔
        colsInfos['std_WinIdx'] = [] # 标准窗列表：每列每窗对应的标准窗编号
        # 1) 初步更新：仅根据当前列补充列信息
        for colID in range(colsInfos['col_num']):  # 列编号
            col_WinIDs = colsInfos['colWinIdxs'][colID]  # 当前列的窗户编号
            if len(col_WinIDs) <= 2:
                winEve = windowsNP[col_WinIDs[0]]
                if len(col_WinIDs)>1:  # 两个求均值
                    winEve = (winEve + windowsNP[col_WinIDs[1]])/2
                std_size = [winEve[2] - winEve[0], winEve[3] - winEve[1]]  # W,H
                std_center = [(winEve[2] + winEve[0]) / 2, (winEve[3] + winEve[1]) / 2]  # W,H
                std_dis = 0 if len(col_WinIDs)==1 else  10  #[临时值]
                colsInfos['std_size'].append(std_size)
                colsInfos['std_center'].append(std_center)
                colsInfos['std_dis'].append(std_dis)
            else:
                # 获取两两检索表
                searchID = [[i, j] for i in col_WinIDs for j in col_WinIDs if i < j]
                seatchID2 = [[i, j] for i in col_WinIDs for j in col_WinIDs if i > j]
                # 获取窗宽度列表
                width_rates = confMtx_pos[np.array(searchID)[:,0], np.array(searchID)[:,1]]  # i<j 表示宽度
                i, j = searchID[np.argmax(width_rates)]  # 对应的最大值时【i，j】
                win_1 = (windowsNP[i]+windowsNP[j])/2
                # 获取窗尺寸列表
                confMtx_size = ConfMtx_size(windows)  # 临时 尺寸的混淆矩阵[i>j存储h比例，i<j存储w比例]
                height_rates = confMtx_size[np.array(seatchID2)[:,0], np.array(seatchID2)[:,1]]   # i>j 表示高度
                i, j = seatchID2[np.argmax(height_rates)]  # 对应的最大值时【i，j】
                win_2 = (windowsNP[i] + windowsNP[j]) / 2
                # 计算基础值
                std_size = [win_1[2] - win_1[0], win_2[3] - win_2[1]]  # W,H
                std_center = [(win_1[2] + win_1[0]) / 2, (win_2[3] + win_2[1]) / 2]  # W,H
                colsInfos['std_size'].append(std_size)
                colsInfos['std_center'].append(std_center)
                # 计算窗间距
                disList = [windowsNP[col_WinIDs[numi+1]][1]-windowsNP[col_WinIDs[numi]][3] for numi in range(len(col_WinIDs)-1)]  # 所有窗间距
                std_dis = min(disList)  # 临时
                colsInfos['std_dis'].append(std_dis)
        # 更新标准窗编号
        colsInfos['std_WinIdx'] = copy.deepcopy(colsInfos['colWinIdxs']) # 标准窗列表：每列每窗对应的标准窗编号
        for colsi in range(len(colsInfos['std_WinIdx'])):
            for winidi in range(len(colsInfos['std_WinIdx'][colsi])):  # 逐个窗判断
                searchid = colsInfos['colWinIdxs'][colsi][winidi]
                colsInfos['std_WinIdx'][colsi][winidi] = -1  # 非标准窗值为-1
                # 判断当前窗在标准窗中的类别
                for stdNum in range(len(stdWinsInfo['list'])):
                    if searchid in stdWinsInfo['list'][stdNum]:
                        colsInfos['std_WinIdx'][colsi][winidi] = stdNum
                        break
        # 2）二次更新：根据列匹配更新
        # 模板窗优化：当前列匹配窗太差，则从匹配列中查找模板窗
        test = 1

        # 5 列匹配
        def sameCol(coli, colj, thresh):
            '''
            列匹配, 列方向存在三个以上标准窗位置匹配，即当作匹配列
            :param coli: 该列窗户编号
            :param colj: 该列窗户编号
            :param thresh: 判断阈值
            :return:
            '''
            num = 0  # 记录列方向重叠度超过阈值的窗户数量
            # 两列窗编号两两匹配
            pairWin = [[i, j] for i in coli for j in colj]
            pairWin = np.sort(np.array(pairWin), axis=1)  # 因为confMtx_pos的一二维索引，只有1维>2维才表示h的iou
            for ij in pairWin:  # 【可优化】列表索引取出所有需要的值，统计合格数量
                if confMtx_pos[ij[1]][ij[0]] > thresh:  # 列方向相关性  # 注意区分一二维序号
                    num += 1
            if num >= 3:
                return True
            else:
                return False

        colNum = colsInfos['col_num']  # len(colsInfos['colWinIdxs'])  # 列数量
        confMtx_ColMatchBool = np.zeros([colNum, colNum], dtype=bool)  # 同列的混淆矩阵 bool
        for i in range(colNum):
            for j in range(colNum):
                # 去除两列
                if i == j:
                    continue
                # 不处理-标签为后续不处理的列
                if colsInfos['process'][i] == False or colsInfos['process'][j] == False:
                    continue
                coli, colj = colsInfos['colWinIdxs'][i], colsInfos['colWinIdxs'][j]
                if sameCol(coli, colj, 0.8):
                    confMtx_ColMatchBool[i][j] = True
                    confMtx_ColMatchBool[j][i] = True


        # 深度优先搜索构建列关系
        colrelation = initColList(confMtx_ColMatchBool)  # 列序列，如[[1,2,3],[4,5,6]]
        colsInfos['col_group'] = colrelation  # 标准列列表

        # 6 列补充
        # 6.1 水平匹配构造标准列
        addWins = winStandard3(colsInfos, windows, stdWinsInfo)
        # 6.2 门重叠 排除部分窗户
        if True:
            # if wall['name'] != '0AJe6aXIV6Oq_61':
            #     continue
            addWins_NOdoor = []
            for win in addWins:
                center_win = [(win[2] + win[0]) / 2, (win[3] + win[1]) / 2]
                size_win = [win[2] - win[0], win[3] - win[1]]
                winCross = False  # 是否与门相交
                for door in doors:
                    # 如果两个矩形中心坐标差值超过两边和，说明相交
                    center_door = [(door[2] + door[0]) / 2, (door[3] + door[1]) / 2]
                    size_door = [door[2] - door[0], door[3] - door[1]]
                    if abs(center_door[0] - center_win[0]) < (size_door[0] + size_win[0]) / 2 \
                            and abs(center_door[1] - center_win[1]) < (size_door[1] + size_win[1]) / 2:  # 相交
                        winCross = True
                        break
                if not winCross:
                    addWins_NOdoor.append(win)
            addWins = addWins_NOdoor


        # 测试
        if testMode:
            path, name = wall['path'], wall['name']
            img = cv2.imread(path)
            pic_h, pic_w, _ = img.shape

            # if name !='5yYSFSckVlhc_13':
            #     continue
            # -1) 门
            if False:
                for door in doors:
                    rect = door
                    rect = list(map(int, rect))
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), int(pic_h / 200))
            # 0) 保留的分割
            if True:
                ids = winIdx_dels | winIdx_refreshs
                for winid in range(len(windows)):
                    if winid in ids:  # 不显示剔除窗
                        continue
                    rect = windows[winid]
                    rect = list(map(int, rect))
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), int(pic_h / 350))
            # 1) 标准窗列表
            if False:
                StandWinIdxs = stdWinsInfo['list']
                for groupi in StandWinIdxs:
                    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    for winidx in groupi:
                        rect = windowsNP[winidx]
                        rect = list(map(int, rect))
                        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (color), int(pic_h/200))
            # 2) 显示列
            if False:
                for wincolIdx in windowColIdxs:
                    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    wincol = windowsNP[wincolIdx]  # 当前列窗户
                    winCol_Min = np.min(wincol[:,:2], axis=0)  # 左上点获取最小值
                    winCol_Max = np.max(wincol[:,2:], axis=0)  # 右下点获取最大值
                    rect =np.hstack([winCol_Min, winCol_Max])
                    rect = list(map(int, rect))
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (color), int(pic_h/250))

            # 3) 列优化:窗户替换
            if True:
                # if len(winIdx_news) == 0:  # 不显示无修改窗
                #     continue
                # for win in winIdx_dels:  # 删除窗
                #     rect = windows[win]
                #     rect = list(map(int, rect))
                #     cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), int(pic_h / 350))
                for id, win in winIdx_news.items():  # 新增窗
                    rect = win
                    rect = list(map(int, rect))
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), int(pic_h / 350))

            # 4) 列的模板窗
            if False:
                for i in range(colsInfos['col_num']):
                    centeri = colsInfos['std_center'][i]
                    whi = colsInfos['std_size'][i]
                    rect = [centeri[0]-whi[0]/2, centeri[1]-whi[1]/2, centeri[0]+whi[0]/2, centeri[1]+whi[1]/2]
                    rect = list(map(int, rect))
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255,255, 0), int(pic_h / 200))

            # 4） 列匹配
            if False:
                # numGroup = len(colrelation)
                # colori=0
                for collisti in colrelation:  # collisti 匹配同组的列
                    if len(collisti)<2:  # 少于两个列
                        continue
                    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    for colIdx in collisti:  # colIdx列编号
                        wincolIdx = colsInfos['colWinIdxs'][colIdx]  # 获取当前列的窗编号
                        wincol = windowsNP[wincolIdx]  # 当前列窗户
                        winCol_Min = np.min(wincol[:, :2], axis=0)  # 左上点获取最小值
                        winCol_Max = np.max(wincol[:, 2:], axis=0)  # 右下点获取最大值
                        rect = np.hstack([winCol_Min, winCol_Max])
                        rect = list(map(int, rect))
                        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (color), int(pic_h / 500))
            # 5) 列推理补充窗户
            if True:
                # if addWins==[]:
                #     continue
                for rect in addWins:
                    rect = list(map(int, rect))
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), int(pic_h / 300))

            if False: #showBool=True
                img_show_W = 1000
                img_zip = cv2.resize(img, (img_show_W, int(img_show_W * pic_h / pic_w)), interpolation=cv2.INTER_AREA)
                cv2.imshow(name, img_zip)
                cv2.waitKey(0)  # 等待按键按下
                cv2.destroyAllWindows()  # 关闭所有窗口
            if True:  # writeBool=True
                cv2.imwrite(os.path.join(tempPath, name + '_temp.jpg'), img)

        # 7 返回需要剔除的窗户-原始下标
        # 1）
        winIdx_dels = list(winIdx_dels)
        idx_ORG_del = idx_ORG[winIdx_dels]  # 删除的窗户(原始窗户顺序)
        # 2）
        winIdx_refreshs = list(winIdx_refreshs)
        idx_ORG_Change = idx_ORG[winIdx_refreshs]
        # 3）合并窗户
        addWins.extend(win_changed)  # 二维窗户list
        wall['ifoChange'] = {'idx_del':idx_ORG_del.tolist(), 'idx_change':idx_ORG_Change.tolist(), 'addWins':addWins}

    return

def refreshJson2(layoutInputAll, inputJson):
    '''
    brief 将新增的窗户写入json字符串中
    :param layoutInputAll: 更新后的布局
    :param inputJson: json字符串
    :return:
    '''
    # wall['ifoChange'] = {'idx_del': idx_ORG_del, 'idx_change': idx_ORG_Change, 'addWins': addWins}

    for wall in layoutInputAll:
        name = wall['name']
        if name != '0AJe6aXIV6Oq_61':   # test
            # continue
            test = 1
        wins = wall['window']
        if 'ifoChange' not in wall:
            print(f"没有修改{name}")
            continue
        ifoChange = wall['ifoChange']
        jsonInfo = inputJson[name]  # json中对应数据

        # 1 处理修改的序号
        idx_change = ifoChange['idx_change']
        for idx_change in idx_change:  # 逐个窗户
            # 1) 添加修改后窗户
            # win = wins[idx_change]
            # segmentation = [[win[0], win[1]], [win[0], win[3]], [win[2], win[3]], [win[2], win[1]]]
            # segmentation = [[int(a) for a in sublist] for sublist in segmentation]
            # window = {'category': 'window', 'bbox': win, 'segmentation': segmentation}
            # jsonInfo.append(window)
            # 2) 删除已修改窗户
            ifoChange['idx_del'].append(idx_change)

        # 2 处理新增窗户
        for win in ifoChange['addWins']:  # 逐个窗户
            segmentation = [[win[0], win[1]], [win[0], win[3]], [win[2], win[3]], [win[2], win[1]]]
            segmentation = [[int(a) for a in sublist]for sublist in segmentation]
            window = {'category': 'window', 'bbox': win, 'segmentation': segmentation}
            jsonInfo.append(window)

        # 3 处理删除的序号
        idx_del = ifoChange['idx_del']
        idx_del = np.array(wall['windowIdx'])[idx_del]  # json中的顺序混合了门,需要重新定位
        winsNew = [jsonInfo[item] for item in range(len(jsonInfo)) if item not in idx_del]
        inputJson[name] = winsNew

    return

def layoutResaoning(inputJson,inputImagesFolder, tempFolder, testMode):
    '''
    brief 门窗推理主函数
    :param inputJson:
    :param inputImagesFolder:
    :param tempFolder: 临时文件夹
    :param testMode: 是否输出中间文件
    :return:
    '''
    # 1 初始布局解析: 中间数据放在layoutInputAll中,用于处理
    layoutInputAll = initLayout(inputJson, inputImagesFolder, os.path.join(tempFolder,'0_reco'), False)
    # 门窗类别筛选合并
    # 2 布局推理
    addWindow2(layoutInputAll, tempFolder, testMode)
    # 3 更新json
    refreshJson2(layoutInputAll, inputJson)
    return

if __name__ == '__main__':

    # 输入参数
    testMode = True  # 是否输出中间文件
    # 图片所在文件夹
    inputImagesFolder = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\image'
    # json文件地址
    inputJsonPath = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\result_50.json'  # result_50
    # 临时文件夹(不写出可不给)
    tempFolder = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\temp50'
    outputJsonPath = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\results3out.json'
    # 解析json
    with open(inputJsonPath, 'r', encoding='utf-8') as f:
        inputJson = json.load(f)

    # 主函数
    layoutResaoning(inputJson, inputImagesFolder, tempFolder, testMode)

    # 写出Json
    with open(outputJsonPath, "w") as f:
        json.dump(inputJson, f)