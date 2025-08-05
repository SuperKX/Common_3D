'''
    二维几何运算函数
        poly2bbox       轮廓线转bbox
        polys2bboxs     轮廓线批量转bbox
        bbox2poly       bbox转轮廓线
        rects_relation  判断两个矩形关系
        lines_relation  判断两个线段关系
'''


# 轮廓线转bbox
def poly2bbox(list2d):
    '''
    轮廓线转bbox
    :param list2d: [[x,y],...]
    :return: bbox:[x_min,ymin,z_max,y_max]
    '''
    columns = list(zip(*list2d))
    max_values = [max(col) for col in columns]
    min_values = [min(col) for col in columns]
    # 计算包围盒
    bbox = [min_values[0], min_values[1],max_values[0], max_values[1]]
    return bbox

# 轮廓线批量转bbox
def polys2bboxs(list3d):
    '''
    轮廓线计算包围盒，批量
    :param list3d: list[]
    :return: bboxs：list[]
    '''
    bboxs = []
    for list2d in list3d:
        bboxs.append(poly2bbox(list2d))
    return bboxs

# bbox转轮廓线
def bbox2poly(bbox):
    '''
    bbox转轮廓线
    :param bbox:[x_min,ymin,z_max,y_max]
    :return: list2d: [[x,y],...]
    '''
    poly = [[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
    return poly

# 计算交集面积
def rect_intersection_area(rect1, rect2):
    """
    计算两个矩形的交集面积。
    :param rect1: 第一个矩形的坐标 (x1, y1, x2, y2)
    :param rect2: 第二个矩形的坐标 (x1, y1, x2, y2)
    :return: 交集面积，如果两个矩形不相交，则返回 0
    """
    # 解包矩形坐标
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    # 判断相交
    if (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
        return 0
    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # 计算交集面积
    width_inter = x2_inter - x1_inter
    height_inter = y2_inter - y1_inter
    area_inter = width_inter * height_inter
    return area_inter


def lines_relation(segment1, segment2):
    """
    计算两个线段的占比和交并比（IoU）。
    :param segment1: 第一个线段的坐标 (start1, end1)
    :param segment2: 第二个线段的坐标 (start2, end2)
    :return: 一个元组 (iou, ratio1, ratio2)，其中 ratio1 和 ratio2 分别是交集长度占 segment1 和 segment2 的比例
    """
    start1, end1 = segment1
    start2, end2 = segment2

    # 计算交集长度
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_length = max(0, intersection_end - intersection_start)

    # 计算每个线段的长度
    length1 = end1 - start1
    length2 = end2 - start2

    # 计算占比
    ratio1 = intersection_length / length1 if length1 > 0 else 0
    ratio2 = intersection_length / length2 if length2 > 0 else 0

    # 计算 IoU
    iou = intersection_length / (length1 + length2 - intersection_length) if (length1 + length2 - intersection_length) > 0 else 0

    return iou, ratio1, ratio2


# 判断两个矩形关系
def rects_relation(rect1, rect2, threshold):
    """
    判断两个矩形关系：
         1）相离
         2）相交
         3）阈值范围内相邻接
    :param rect1: 第一个矩形的坐标 (x1, y1, x2, y2)
    :param rect2: 第二个矩形的坐标 (x1, y1, x2, y2)
    :param threshold: 邻接的距离阈值
    :return:
         1）'disjoint'       相离
         2）'intersect',rate1,rate2,iou         相交，交集在两个矩形中的占比，及iou
         3）'adjacent', 'horizontal' 或 'vertical'    邻接关系

    返回两个矩形的关系，可能的值为 'intersect', 'disjoint', 'adjacent'。
             如果是 'adjacent'，还会返回具体的邻接方式，可能的值为 'horizontal' 或 'vertical'。
    """
    # 解包矩形坐标
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    # 计算两个矩形的边界
    left1, right1, top1, bottom1 = x1_1, x2_1, y1_1, y2_1
    left2, right2, top2, bottom2 = x1_2, x2_2, y1_2, y2_2

    # 1 判断邻接
    # 计算两个矩形之间的最小距离
    dx = min(abs(right1 - left2), abs(left1 - right2))
    dy = min(abs(bottom1 - top2), abs(top1 - bottom2))
    # 检查是否邻接
    if dx <= threshold or dy <= threshold:
        if dx <= threshold:
            if min(bottom1,bottom2) - max(top1, top2) > -threshold:  # 另一方向存在交集
                return 'adjacent', 'horizontal'
            else:
                return 'disjoint', ''

        else:
            if min(right1, right2) - max(left1, left2) > -threshold:  # 另一方向存在交集
                return 'adjacent', 'vertical'
            else:
                return 'disjoint', ''

    # 2 判断相交
    area_inter = rect_intersection_area(rect1, rect2)
    if area_inter > 0:
        # 计算每个矩形的面积
        area_rect1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_rect2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        # 计算交集所占比例
        ratio_rect1 = area_inter / area_rect1
        ratio_rect2 = area_inter / area_rect2
        iou = area_inter / (area_rect1 + area_rect2 - area_inter)
        return 'intersect', ratio_rect1, ratio_rect2, iou
    # 3 判断相离
    else:
        return 'disjoint',''

# 轮廓线简化
# approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
import math
def remove_small_angles(points, min_angle_degrees):
    """
    从轮廓线中剔除折线夹角小于指定阈值的点
    参数:
        points: 二维点列表，[(x1, y1), (x2, y2), ...]
        min_angle_degrees: 最小保留夹角度数

    返回:
        处理后的点列表
    """
    if len(points) < 3:
        return points  # 少于3个点无法形成夹角，直接返回

    min_angle_radians = math.radians(min_angle_degrees)
    result = points.copy()
    i = 1

    is_changed =True  # 记录是否完成遍历
    while is_changed:
        is_changed = False
        while i < len(result) - 1:
            # 获取三个连续点
            p_prev = result[i - 1]
            p_current = result[i]
            p_next = result[i + 1]

            # 计算向量
            vec1 = (p_prev[0] - p_current[0], p_prev[1] - p_current[1])
            vec2 = (p_next[0] - p_current[0], p_next[1] - p_current[1])

            # 计算向量点积
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]

            # 计算向量模长
            norm1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
            norm2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

            # 避免除以零（改进判断）
            if norm1 < 1e-8 or norm2 < 1e-8:
                result.pop(i)  # 移除距离过近的点
                continue

            # 计算余弦值并限制在[-1, 1]范围内
            cos_angle = max(-1.0, min(1.0, dot_product / (norm1 * norm2)))

            # 计算夹角（弧度）
            angle = math.acos(cos_angle)

            # 如果夹角小于阈值，则移除当前点
            if angle < min_angle_radians:
                result.pop(i)  # 删除点后索引不需要递增，因为下一个点会移到当前位置
                is_changed = True
            else:
                i += 1  # 只有未删除点时才递增索引

    # 可选：确保处理后的轮廓仍然有效
    if len(result) < 3:
        return points  # 如果处理后点太少，返回原始轮廓

    return result

def get_segments_in_x_range(contour,range_x):
    '''
    轮廓线按照x方向范围进行拆分
    contour 轮廓线
    range_x 水平方向范围
    '''
    x_min, x_max = range_x[0],range_x[1]
    segments = []  # 记录所有分割线段
    n = len(contour)
    # 遍历所有轮廓线段（包括最后一点到第一点的闭合线段）
    # 1 计算所有线段
    for i in range(n):  # 遍历所有轮廓线段
        p1 = contour[i]
        p2 = contour[(i + 1) % n]  # 闭合轮廓，取模获取下一个点
        # 提取线段两端点的x坐标
        x1, y1 = p1
        x2, y2 = p2
        # 判断线段的x范围是否与目标范围有重叠
        seg_x_min = min(x1, x2)
        seg_x_max = max(x1, x2)

        # 0）快速排除无交集的情况
        if seg_x_max < x_min or seg_x_min > x_max:
            continue

        # 1）竖直直线
        if x1 == x2:
            if y1 == y2:  # 排除重叠点
                continue
            if x_min <= x1 <= x_max:
                segments.append([p1.tolist(), p2.tolist()])
            continue

        # 2）水平线或倾斜
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        # 检查线段是否在x范围内有交点或完全包含
        # 线段在x范围内的起始和结束x坐标
        start_x = max(x_min, seg_x_min)
        end_x = min(x_max, seg_x_max)
        if start_x == end_x:
            continue
        # 计算线段在x范围内的两个端点
        start_y = k * start_x + b
        end_y = k * end_x + b
        # 构建范围内的线段
        if x1 < x2:
            segments.append([(start_x, start_y), (end_x, end_y)])
        else:
            segments.append([(end_x, end_y), (start_x, start_y)])

    semments_num = len(segments)
    # 2 将线段拼接成多条折线
    break_id = []  # 记录多段折线分段处
    polys = []
    for j in range(semments_num):  # 计算分段位置
        p1 = segments[j][1]
        p2 = segments[(j+1)%len(segments)][0]
        if p1[0]-p2[0]>1 or p1[0]-p2[0]<-1 or p1[1]-p2[1]>1 or p1[1]-p2[1]<-1:  # 两个点x、y坐标均不重叠在同一像素内
            break_id.append((j+1)%len(segments))

    if len(break_id) == 0:  # 没有分段
        polyi=[]
        for jj in range(len(segments)):
            polyi.append(list(segments[jj][1]))
        return [polyi]

    for k in range(len(break_id)):  # 将线段组成多个折线段
        rangek = [break_id[k], break_id[(k+1)%len(break_id)]]
        polyi = []
        # 连接的线段编号，如 [9,10,0,1,2]
        rangek = range(rangek[0], rangek[1]) if rangek[0] < rangek[1] else [jj%semments_num for jj in range(rangek[0], rangek[1]+semments_num)]

        for t,jj in enumerate(rangek):
            if t==0:
                polyi.append(list(segments[jj][0]))
            polyi.append(list(segments[jj][1]))
        polys.append(polyi)

    # 3 轮廓线排序-按照高程最小值
    # 按每个轮廓中 y 的最小值排序
    sorted_contours = sorted(
        polys,
        key=lambda contour: min(point[1] for point in contour)
    )

    return sorted_contours

def seg_vertical_poly(poly, angle_degrees=45):
    '''
    将一个连续的poly，根据倾斜角度分割成多段，只保留倾斜度低于angle_degrees角度的轮廓线
    '''
    is_same_poly =True
    polyi=[]
    #1 统计分割位置
    polys = []
    poly_idx_seg = []
    for p, point_p in enumerate(poly): ## 逐个线段
        if p == len(poly) - 1:
            if len(polyi) != 0:
                polys.append(polyi)
            break
        p1 = poly[p]
        p2 = poly[p + 1]
        tanx = abs((p1[1] - p2[1]) / (p1[0] - p2[0] + 1e-5))  # tan与x轴夹角

        if tanx <= math.tan(math.radians(angle_degrees)):  # 水平面夹角小于45度保留，否则删除
            if len(polyi) == 0:
                polyi.append(p1)
            polyi.append(p2)
        else:
            if len(polyi) != 0:
                polys.append(polyi)
                polyi = []
    return polys
