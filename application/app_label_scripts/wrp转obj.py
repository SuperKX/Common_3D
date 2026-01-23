'''
brief：处理用geomagic打完标签后，导出obj点云化等功能。
    流程主要分为以下三个步骤：
        1 wrp 生成obj和和标签文件（obj面片顺序跟wrp文件对不上，但顶点列表一致）
            ① wrp转obj文件。
            ② label.pth 生成记录各类别面片编号的pth文件。
            ③ faces_old.pth 生成label.pth对应的面片序列索引
        2 生成labeled_obj文件
            ① faces_new 读入obj文件，获取当前的面片顺序
            ② label_new 将老的类别对应面片编号列表，映射到obj的新面片列表
            ③ obj_labeled 将标签信息写入到obj group中，得到校正后的obj文件。
        3 obj_labeled文件解析
            ① mesh按照group分块、
            ② 点云化处理。
        4 点云文件批量采样及合并
'''
import os
import open3d as o3d
import numpy as np
import pickle
import points.points_io as pts_io
import points.points_common as points_common
import mesh.mesh_common as mesh_fuc
import time
import efficient.parallel as parallel
import efficient.ndarray_calculate as ndarray_calculate

def read_dict_to_binary(filename):
    '''
    解析 标签.pth格式的字典，
    注意：
        1）wrp中标签下标从1开始，此处-1，跟面片下标对齐。
        2）默认每个标签的面片先排序。
    '''
    data = dict()
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print("success read")
    except Exception as e:
        print("wrong read!")
    # 排序，优化
    for key, val in data.items():  # wigwam
        # val.sort()
        val = (np.array(val) - 1).tolist()  # 编号从1开始
        data[key] = val
    return data

def read_mesh_verts_idxs(filename):
    '''
    读入pth文件，解析为ndarray(n,3)，按照顺序记录一系列面片的顶点索引，如[[10 4 7],[5,6,90],...]
    索引改为 从0开始！
    '''
    import pickle
    data = dict()
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print("success read")
    except Exception as e:
        print("wrong read!")
    output= np.array(data)-1  # 索引从1开始
    return output

def read_obj_minimal(filename):
    """
    obj文件解析，按照文件顺序，返回面片列表、纹理对应的面片字典,mtl 出现的行号
    Args:
        filename (str): Path to the OBJ file.
    Returns:
        face_list_new (list): 新的标签列表。ex：[[v1,v2,v3],[...],...]
        material_to_faces (dict): 纹理贴图到到新面片序列的列表，{‘mtl1':[f1,f2,f3,...],...}
        mtl_file_line_num: 记录mtl文件出现的行号
    """
    face_list_new = []
    material_to_faces = {}
    current_material = None
    mtl_file_line_num = -1

    with open(filename, "r") as f:
        for line_number, line in enumerate(f):  # Capture line number
            line = line.strip()

            if line.startswith("usemtl "):
                # Use material
                current_material = line[7:]
                if current_material not in material_to_faces:
                    material_to_faces[current_material] = {} # Change to nested dict
            elif line.startswith("f "):
                # Face indices
                face_data = line[2:].split()
                face = []
                for vertex_data in face_data:
                    # Extract only the vertex index (first component)
                    vertex_index = int(vertex_data.split("/")[0]) - 1
                    face.append(vertex_index)

                face_list_new_index = len(face_list_new) # Get index BEFORE appending

                face_list_new.append(face)  # Add to the new list

                # Add face index in face_list_new and line content to the current material's list
                if current_material:
                    material_to_faces[current_material][face_list_new_index] = line # Store face_list_new_index and line content
            elif line.startswith("mtllib "):
                mtl_file_line_num =line_number
    return face_list_new, material_to_faces,mtl_file_line_num


def create_selected_material_dict(material_to_faces, face_list_selected):
    """
    创建材质到选中面片内容的映射字典
    （优化、加速）
    优化版本特点：
    - 预先将face_list_selected转为集合实现O(1)查找
    - 使用列表推导式替代extend操作
    - 无匹配材质时提前跳过处理

    参数：
        material_to_faces (dict): 材质字典，结构为{材质名: {面片索引: 内容,...}}
        face_list_selected (list/array): 需要包含的选中面片索引列表

    返回：
        dict: 结构为{材质名: [对应内容]}，仅包含选中面片
    """
    # 转换为集合实现快速查找
    selected_set = set(face_list_selected)
    return_dict = {}

    for material_name, face_info in material_to_faces.items():
        # 使用集合交集运算（比np.intersect1d更快）
        common_faces = selected_set & face_info.keys()
        if not common_faces:
            continue
        # 直接通过列表推导式构建结果
        return_dict[material_name] = [face_info[face] for face in common_faces]
    return dict(return_dict)  # 转回普通字典类型

def replace_file_content(filename, start_line, new_content):
    """
    指定行，替换字符串。
    Args:
        filename (str): The path to the file.
        start_line (int): The line number (1-based) from which to start replacing.
        new_content (str): The new content to write to the file.
    """

    try:
        with open(filename, "r") as f:
            lines = f.readlines()  # Read all lines into a list

        # Validate start_line
        if not (1 <= start_line <= len(lines) + 1):  #Line 1 is the first line
            raise ValueError(f"start_line must be between 1 and {len(lines) + 1}")


        # Modify the lines list. index  = line_number -1
        if start_line <= len(lines):
             lines[start_line-1:] = new_content.splitlines(keepends=True)  #Splits the content into lines, preserves line endings
        else: # append
             lines.extend(new_content.splitlines(keepends=True))

        with open(filename, "w") as f:
            f.writelines(lines)  # Write the modified lines back to the file

        print(f"Successfully replaced content in {filename} starting from line {start_line}.")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except ValueError as e:
        print(f"Error: Invalid start_line: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def label_obj(labels_new, classlist,material_to_faces, obj_file,start_line):
    '''
        将标签打入obj文件中
        当前版本：创建同名的'_group'版本

        注意：
            1）当前默认写出到'_group.obj'文件中
       '''
    # 1 创建新文件，复制必须部分
    import file.file_common as file_co
    input_file = obj_file
    output_file = os.path.splitext(obj_file)[0]+"_group"+os.path.splitext(obj_file)[1]
    file_co.copy_first_n_lines(input_file, output_file, start_line+1)  # 前n行拷贝

    # 2 分group
    # face_str = ''  # 用于记录更新后面片的字符信息.
    for key, face_list_selected in labels_new.items():  # 'wigwam'
        if key not in classlist:
            print(f"【注意】：该标签 {key} 未统计！")
        if len(face_list_selected) <= 0:
            continue
        # 2) 获取
        group_dic = create_selected_material_dict(material_to_faces, face_list_selected)
        lines = [f'g {key}']
        for mtl_name, mtl_infor in group_dic.items():
            lines.append(f'usemtl {mtl_name}')
            lines.extend(mtl_infor)
        group_str = '\n'.join(lines) + '\n'  # 记录当前类别相关所有字符串
        with open(output_file, 'a') as f:  # 追加模式
            f.write(group_str)
        print(f"    完成标签  {key} 在obj中的分组")
    # start_line=61870
    # replace_file_content(obj_file, start_line + 2, face_str)  # 编号删除
    # raise ValueError("结束！")


def mesh_to_group(filename, pth_fileP, obj_file, dic_old_pth):
    '''
    将mesh按照group分块。

        filename        文件名
        pth_fileP       打标签的pth文件
        inputPath_obj   /obj的目录（obj文件、老序列的pth）
    注意：
        1） 默认老面片序列从obj同目录获取
    '''
    start_time = time.time()
    print(f"{filename} 2 obj打标签开始")
    if not os.path.exists(obj_file):
        raise ValueError(f"{obj_file} 不存在！")
    # 解析 标签.pth
    if not os.path.exists(pth_fileP):
        raise ValueError(f"{pth_fileP} 不存在！")
    class_labels = read_dict_to_binary(pth_fileP)
    # 解析 标签对应的面片序列.pth
    # 老序列
    if not os.path.exists(dic_old_pth):
        raise ValueError(f"{dic_old_pth} 不存在！")
    face_list_old = read_mesh_verts_idxs(dic_old_pth)
    print(f"{filename} 2.1 完成 old序列 pth解析 {(time.time() - start_time):.4f}")
    start_time = time.time()
    # 2）标签新序列
    face_list_new, material_to_faces, mtl_file_line_num = read_obj_minimal(obj_file)  # 所有面索引
    face_list_new = np.array(face_list_new)
    print(f"{filename} 2.2 完成 new序列 解析 {(time.time() - start_time):.4f}")
    start_time = time.time()
    # 建立新旧面片顺序的映射
    new_old_idx = ndarray_calculate.create_index_array(face_list_old, face_list_new)  # 索引,记录旧的映射到新的
    print(f"{filename} 2.3 完成 新旧面索引映射 {(time.time() - start_time):.4f}")
    start_time = time.time()
    # 更新标签列表
    labels_new = dict()
    for key, face_list in class_labels.items():  # wigwam
        labels_new[key] = new_old_idx[face_list]
    # 3) 更新打标签后的obj文件
    label_obj(labels_new, classlist, material_to_faces, obj_file, mtl_file_line_num)
    print(f"{filename} 2 完成obj打标签 {(time.time() - start_time):.4f}")
    start_time = time.time()


def groupedobj_to_cloud(filename, obj_file_grouped, subdense, cloud_file, bool_write_ply=True):
    '''
        单个obj文件点云化
        filename    文件名
        obj_file    obj文件地址
        subdense    采样密度
        cloud_file  写出地址
        bool_write_ply  是否写出

        return merged_cloud 合并后点云
    '''
    # 3 obj文件分割
    print(f"{filename} 3 开始mesh点云化处理")
    start_time = time.time()
    # TODO 考虑替换文件，命名通一等，待优化！！！！！！！！！！！！！！
    obj_file = obj_file_grouped
    meshes = mesh_fuc.crop_mesh_by_group(obj_file)
    print(f"{filename} 3.1 完成mesh分块 {(time.time() - start_time):.4f}")
    start_time = time.time()
    # 4 mesh点云化
    merged_cloud = np.empty((0, 7))
    for name, mesh in meshes.items():
        cloud_seg = mesh_fuc.mesh_2_points(mesh, subdense, min_point_Num=1)
        if len(cloud_seg.colors) <= 0:
            print(f"没有颜色，查看是否有贴图！")
        # 1） 构造np格式点云(打标签单独封装功能)
        coord_seg = np.hstack((np.array(cloud_seg.points), np.array(cloud_seg.colors) * 255))
        lablei = label_list[classlist.index(name)]
        points_seg_np = np.hstack(
            (coord_seg, np.full((coord_seg.shape[0], 1), lablei)))  # x,y,z, r,g,b, label
        # 5 分块点云合并
        merged_cloud = np.vstack((merged_cloud, points_seg_np))  # shape(N,7)
    print(f"{filename} 3 完成点云化 {(time.time() - start_time):.4f}")
    # 写出点云
    pts_io.write_ply_file_binary_batch(cloud_file, merged_cloud)
    return merged_cloud

# 当前弃用，仅作为备份参考，使用whole_process_parallel替代。2025.07.29
# 没有维护
def whole_process(path_env, generate_labeled_obj=True, generate_all_clouds=True, generate_merged_cloud=True):
    '''
    全流程处理
        path_env    输入场景的地址（内部结构："\obj"，"\pth"），其中根据pth文件索引遍历
        generate_labeled_obj    生成grouped的obj文件
        generate_all_clouds     obj分块点云化
        generate_merged_cloud   点云降采样及合并
    '''
    time_start = time.time()
    #0  配置信息
    # 1）标签(已经改为全局)
    # 2）地址
    inputPath_obj = path_env + r"\obj"
    inputPath_pth = path_env + r"\pth"
    outputPath = path_env + r"\ply"
    # 3）采样参数
    min_point_Num = 1  # 分块保证的最小采样点数量
    subdense = 0.1  # gridsample密度
    # 4) 临时
    all_clouds = []  # 采样后点云

    # 主函数
    if not os.path.exists(inputPath_obj) or not os.path.exists(inputPath_pth):
        raise FileNotFoundError("找不到输入路径！")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for pth_file in os.listdir(inputPath_pth):
        start_time = time.time()
        # 0) 处理地址
        filename, ext = os.path.splitext(pth_file)
        if ext != '.pth':
            continue
        obj_file = os.path.join(inputPath_obj, filename, filename + '.obj')  # 读入的obj文件
        pth_fileP = os.path.join(inputPath_pth, pth_file)  # 读入的标签文件
        # 点云写出地址
        cloud_file = os.path.join(outputPath, filename + '.ply')
        # 1 wrp处理，放到"wrap脚本.py"(备份在同目录下文件wrap_2_label.py)中处理，需要在geomagic环境下运行。
        # 2 生成labeled_obj文件
        if generate_labeled_obj:
            obj_file = os.path.join(inputPath_obj, filename, filename + '.obj')  # 读入的obj文件
            dic_old_pth = os.path.join(inputPath_obj, filename, filename + '_old_face_dict.pth')
            mesh_to_group(filename, pth_fileP, obj_file, dic_old_pth)

        # 3 点云分割
        if generate_all_clouds:
            obj_file_grouped = os.path.splitext(obj_file)[0] + "_group" + os.path.splitext(obj_file)[1]
            merged_cloud = groupedobj_to_cloud(filename, obj_file_grouped, subdense, cloud_file, bool_write_ply=True)

        # 4 点云采样
        if generate_merged_cloud and generate_all_clouds:  # 全局处理点云合并的时候，需要点云化过程，为了节省io时间。
            start_time = time.time()
            cloud = points_common.gridsample_np_points(merged_cloud, voxel_size=0.1)
            all_clouds.append(cloud)
            print(f"{filename} 4 完成点云采样 {(time.time() - start_time):.4f}")

    # 4 点云合并及写出
    if generate_merged_cloud and generate_all_clouds:
        start_time = time.time()
        merged_points = np.vstack(all_clouds)
        # 点云写出
        # import pts_io
        output_file = os.path.join(path_env, filename + '_sub0.1.ply')
        pts_io.write_ply_file_binary_batch(output_file, merged_points)
        print(f"5 完成点云合并及写出 {(time.time() - start_time):.4f}")
    print(f"总耗时： {(time.time() - time_start):.4f}")

def whole_process_parallel(path_env, generate_labeled_obj=True, generate_all_clouds=True, generate_merged_cloud=True):
    '''
    全流程处理
        path_env    输入场景的地址
        generate_labeled_obj    生成grouped的obj文件
        generate_all_clouds     obj分块点云化
        generate_merged_cloud   点云降采样及合并
    '''
    time_start = time.time()
    log_str=''
    # 设置参数
    min_point_Num = 1  # 分块保证的最小采样点数量
    subdense = 0.1  # gridsample密度

    # 0 构造传入参数
    scene_name = os.path.basename(path_env)
    inputPath_obj = path_env + r"\obj"
    inputPath_pth = path_env + r"\pth"
    outputPath = path_env + r"\ply"
    if not os.path.exists(inputPath_obj) or not os.path.exists(inputPath_pth):
        raise FileNotFoundError("找不到输入路径！")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    paras = {'filename':[],
             'obj_file':[],
             'pth_fileP': [],
             'cloud_file': [],
             'dic_old_pth': [],
             'obj_file_grouped': []
             }
    for pth_file in os.listdir(inputPath_pth):
        # 处理地址
        filename, ext = os.path.splitext(pth_file)
        if ext != '.pth':
            continue
        obj_file = os.path.join(inputPath_obj, filename, filename + '.obj')  # 读入的obj文件
        pth_fileP = os.path.join(inputPath_pth, pth_file)  # 读入的标签文件
        cloud_file = os.path.join(outputPath, filename + '.ply')        # 点云写出地址
        dic_old_pth = os.path.join(inputPath_obj, filename, filename + '_old_face_dict.pth')  # 老标签地址
        obj_file_grouped = os.path.splitext(obj_file)[0] + "_group" + os.path.splitext(obj_file)[1]
        # 构建
        paras['filename'].append(filename)
        paras['obj_file'].append(obj_file)
        paras['pth_fileP'].append(pth_fileP)
        paras['cloud_file'].append(cloud_file)
        paras['dic_old_pth'].append(dic_old_pth)
        paras['obj_file_grouped'].append(obj_file_grouped)

    # parallel执行功能
    # 1 生成grouped - obj文件
    if generate_labeled_obj:
        start_time = time.time()
        args = [(paras['filename'][item], paras['pth_fileP'][item],paras['obj_file'][item],paras['dic_old_pth'][item])
                for item in range(len(paras['filename']))]
        parallel.parallel_process(mesh_to_group, args)
        log_str += f" 1 生成grouped - obj文件： {(time.time() - start_time):.4f}\n"

    # 2 mesh点云化
    if generate_all_clouds:
        start_time = time.time()
        args = [(paras['filename'][item], paras['obj_file_grouped'][item], subdense, paras['cloud_file'][item], True)
                for item in range(len(paras['filename']))]
        merged_clouds = parallel.parallel_process(groupedobj_to_cloud, args)
        log_str += f"2 mesh点云化： {(time.time() - start_time):.4f}\n"

    # 3 点云采样
    if generate_merged_cloud and generate_all_clouds:  # 全局处理点云合并的时候，需要点云化过程，为了节省io时间
        start_time = time.time()
        voxel_size = 0.1
        args = [(merged_clouds[item], voxel_size) for item in range(len(paras['filename']))]
        subsample_clouds = parallel.parallel_process(points_common.gridsample_np_points, args)
        log_str += f"3 点云采样： {(time.time() - start_time):.4f}\n"

        # 4 点云合并及写出
        start_time = time.time()
        scene_cloud = np.vstack(subsample_clouds)
        output_file = os.path.join(path_env, scene_name + '.ply')
        pts_io.write_ply_file_binary_batch(output_file, scene_cloud)
        log_str += f" 4 点云合并及写出： {(time.time() - start_time):.4f}\n"
    log_str += f" 总耗时： {(time.time() - time_start):.4f}\n"
    print(log_str)

def get_all_files(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def cloud_read_sample(file):
    # 1 读入
    coords, colors, labels = pts_io.parse_ply_file(file)
    if len(labels) == 0:
        raise ValueError("【错误】没有标签")
    # 创建包含所有数据的数组
    data = np.hstack((coords, colors, labels[:, np.newaxis]))
    # 2 采样
    cloud = points_common.gridsample_np_points(data, voxel_size=0.1)
    return cloud

if __name__ == '__main__':

# 参数设置区----------------------------------------------
    global classlist
    classlist = ["background", "building", "wigwam", "car", "vegetation", "farmland",
                 "shed", "stockpiles", "bridge", "pole", "others", "grass"]
    global label_list
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 功能执行区----------------------------------------------
    # 0 全流程处理
    if False:
        time_start = time.time()
        path_env = r"J:\tempdata\temp_process_20251014\27DATA19_segs2"  # "J:\DATASET\BIMTwins\WRP\out\batch1\39PTY2_out"
        # whole_process_parallel(path_env, generate_labeled_obj=True, generate_all_clouds=True, generate_merged_cloud=True)
        # 不并行
        whole_process(path_env, generate_labeled_obj=True, generate_all_clouds=True, generate_merged_cloud=True)

    # 1 mesh按照标签group处理
    # 1.1 单个文件处理
    if True:
        filename = r'Tile_0008_0011'
        folder_path = r'J:\DATASET\BIMTwins\中间数据_标准版_2025.10.15\01JXY'
        pth_fileP = os.path.join(folder_path,'pth', filename+'.pth')
        inputPath_obj = os.path.join(folder_path,'obj')  # 文件夹
        obj_file = os.path.join(inputPath_obj, filename, filename + '.obj')  # 读入的obj文件
        dic_old_pth = os.path.join(inputPath_obj, filename, filename + '_old_face_dict.pth')
        mesh_to_group(filename, pth_fileP, obj_file, dic_old_pth)
    # 1.2 文件夹批量处理

    # 2 grouped_mesh点云化
    # 2.1 单个文件处理
    if False:
        subdense = 0.1
        filename = r'Tile_+001_+002'
        folder_path = r'J:\DATASET\BIMTwins\WRP\out\batch1\34PTY1_out'
        obj_file = os.path.join(folder_path,'obj',filename,filename+'.obj')
        cloud_file = os.path.join(folder_path,'ply',filename+'.ply')  # output
        obj_file_grouped = os.path.splitext(obj_file)[0] + "_group" + os.path.splitext(obj_file)[1]
        merged_cloud = groupedobj_to_cloud(filename, obj_file_grouped, subdense, cloud_file, bool_write_ply=True)
    # 2.2 文件夹批量处理
    if False:
        time_start = time.time()
        obj_folder = r'J:\DATASET\BIMTwins\WRP\out\batch1\34PTY1_out\obj'
        all_files = [d for d in os.listdir(obj_folder)]
        if True:  # 并行 120.8s
            args = []
            for filename in all_files:
                subdense = 0.1
                obj_file = os.path.join(obj_folder, filename, filename + '.obj')
                cloud_file = os.path.join(os.path.split(obj_folder)[0], 'ply', filename + '.ply')  # output
                obj_file_grouped = os.path.splitext(obj_file)[0] + "_group" + os.path.splitext(obj_file)[1]
                args.append((filename, obj_file_grouped, subdense, cloud_file, True))
            parallel.parallel_process(groupedobj_to_cloud, args)
        else:  # 177.1s
            for filename in all_files:
                subdense = 0.1
                obj_file = os.path.join(obj_folder, filename, filename + '.obj')
                cloud_file = os.path.join(os.path.split(obj_folder)[0], 'ply', filename + '.ply')  # output
                obj_file_grouped = os.path.splitext(obj_file)[0] + "_group" + os.path.splitext(obj_file)[1]
                merged_cloud = groupedobj_to_cloud(filename, obj_file_grouped, subdense, cloud_file,
                                                   bool_write_ply=True)

        print(f"2 mesh批量点云化 耗时 {(time.time() - time_start):.4f}")  # 耗时 68


    # 3 点云采样及合并
    if False:
        import threading
        folder_path = r"H:\TempProcess\wrp_process\39PTY2"
        scene_name = os.path.split(folder_path)[1]
        output_file = os.path.join(folder_path, scene_name+'.ply')
        folder_ply = os.path.join(folder_path, 'ply')
        all_files = get_all_files(folder_ply)
        print(f"处理的分块数量: {len(all_files)}")
        # 点云采样
        all_clouds = []
        time0 = time.time()

        if True:  # 并行方法 102s->60s
            input_list = [(item,) for item in all_files]
            all_clouds = parallel.parallel_process(cloud_read_sample,input_list)
        else:  # 无并行，暂不用
            for file in all_files:
                # 1 读入
                coords, colors, labels = pts_io.parse_ply_file(file)
                if len(labels) == 0:
                    raise ValueError("【错误】没有标签")
                # 创建包含所有数据的数组
                data = np.hstack((coords, colors, labels[:, np.newaxis]))
                # 2 采样
                cloud = points_common.gridsample_np_points(data, voxel_size=0.1)
                all_clouds.append(cloud)
                print(f"完成点云{file}")
        print(f"耗时 {(time.time() - time0):.4f}")  # 耗时 68
        # 3 合并
        merged_points = np.vstack(all_clouds)
        # 点云写出
        time0 = time.time()
        pts_io.write_ply_file_binary(output_file, merged_points)
        print(f"耗时 {(time.time() - time0):.4f}")  # 耗时 68
        # 点云写出
        time0 = time.time()
        pts_io.write_ply_file_binary_batch(output_file, merged_points)
        print(f"耗时 {(time.time() - time0):.4f}")  # 耗时 68




