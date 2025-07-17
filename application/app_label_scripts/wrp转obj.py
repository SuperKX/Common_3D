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
'''
import os
import open3d as o3d
import numpy as np
import pickle
# import points.points_io as pts_io
import mesh.mesh_common as mesh_fuc


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
        val.sort()
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
    Reads an OBJ file, returns faces, material-to-faces mapping using index from face_list_new.
    Args:
        filename (str): Path to the OBJ file.
    Returns:
        tuple: (face_list_new, material_to_faces)
            face_list_new (list): 新的标签列表
            material_to_faces (dict): 纹理贴图到到新面片序列的列表
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

def create_index_array(old_indices, new_indices):
    """
    新旧映射，返回1维ndarray，每个位置记录新索引坐标
    Generates a 1D NumPy array mapping old face indices to new face indices.

    Args:
        old_indices (np.ndarray): Shape (n, 3), old face indices.
        new_indices (np.ndarray): Shape (n, 3), new face indices.

    Returns:
        np.ndarray: A 1D NumPy array of shape (n,), where each element
                     represents the new index corresponding to the old index.
                     Returns None if the indices can't be mapped reliably.
                     Unmapped indices are assigned a value of -1.
    """

    if old_indices.shape != new_indices.shape:
        print("Error: The shapes of the old and new indices arrays must be the same.")
        return None

    index_array = np.full(old_indices.shape[0], -1, dtype=int)  # 初始化为 -1

    for i in range(old_indices.shape[0]):
        old_face = old_indices[i]

        # 在 new_indices 中查找相同的面片
        found_indices = np.where((new_indices == old_face).all(axis=1))[0]

        if len(found_indices) == 0:
            print(f"Warning: Face {old_face} not found in new_indices.  Leaving as -1.")
            continue # Leave as -1
            # raise ValueError(f"Face {old_face} not found in new_indices.")
        elif len(found_indices) > 1:
            print(f"Warning: Face {old_face} found multiple times in new_indices.  Using first match, mapping may be ambiguous.")
            # raise ValueError(f"Face {old_face} found multiple times in new_indices.")

        new_index = found_indices[0]  # 获取新索引
        index_array[i] = new_index  # 存储新索引

    return index_array

def create_selected_material_dict(material_to_faces, face_list_selected):
    """
    Creates a dictionary mapping material names to a list of line contents from the selected faces.

    Args:
        material_to_faces (dict): Dictionary mapping material names to a dictionary.
                                  Each material maps to a dict: {face_list_new_index: line_content, ...}
        face_list_selected (list): A list of face indices to include in the output dictionary.

    Returns:
        dict: A dictionary mapping material names to a list of line contents of selected faces
              using that material.
    """
    return_dic = {}
    # Iterate through each material in the material_to_faces dictionary
    for material_name, face_info in material_to_faces.items():
        selected_lines = []

        # Iterate through each face index and line content for the current material
        for face_index, line_content in face_info.items():
            # Check if the current face index is in the face_list_selected
            if face_index in face_list_selected:
                selected_lines.append(line_content) # Append line_content NOT face_index

        # If there are selected lines for the current material, add it to the return_dic
        if selected_lines:
            return_dic[material_name] = selected_lines

    return return_dic

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
       '''
    # 分group
    face_str = ''  # 用于记录更新后面片的字符信息.
    for key, face_list_selected in labels_new.items():  # wigwam
        if key not in classlist:
            print(f"【注意】：该标签 {key} 未统计！")
        if len(face_list_selected) <= 0:
            continue
        # 2) 获取
        group_dic = create_selected_material_dict(material_to_faces, face_list_selected)
        group_str = 'g ' + key + '\n'
        for mtl_name, mtl_infor in group_dic.items():
            group_str += 'usemtl ' + mtl_name + '\n'
            for stri in mtl_infor:
                group_str += stri + '\n'
        # 更新列表
        face_str += group_str
    # start_line=61870
    replace_file_content(obj_file, start_line + 2, face_str)  # 编号删除

def write_ply_file(file_path,cloud_ndarray):
    '''
     临时写出，points_io.py中copy
    '''
    try:
        num_points = cloud_ndarray.shape[0]
        np.savetxt(file_path, cloud_ndarray, fmt='%f %f %f %d %d %d %i')
        # 定义 PLY 文件头
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property int red
property int green
property int blue
property int class
end_header
"""
        with open(file_path, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(header)
            f.write(old)
        print(f"合并后的带标签点云已成功保存到 {file_path}")

    except Exception as e:
        print(f"保存文件时出现错误: {e}")

if __name__ == '__main__':
    #0  配置信息
    # 1）标签
    classlist = ["background", "building", "wigwam", "car", "vegetation", "farmland",
                 "shed", "stockpiles", "bridge", "pole", "others", "grass"]
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # 2）地址
    path_env = r"E:\LabelScripts\testdata\27data19_20250717"
    inputPath_obj = path_env + r"\obj"
    inputPath_pth = path_env + r"\pth"
    outputPath = path_env + r"\ply"
    # 3）采样参数
    min_point_Num = 1  # 分块保证的最小采样点数量
    subdense = 0.1  # gridsample密度

    # 主函数
    if not os.path.exists(inputPath_obj) or not os.path.exists(inputPath_pth):
        raise FileNotFoundError("找不到输入路径！")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for pth_file in os.listdir(inputPath_pth):
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
        print(f"{filename} 2 obj打标签开始")
        # 解析 标签.pth
        class_labels = read_dict_to_binary(pth_fileP)
        # 解析 标签对应的面片序列.pth
        # 老序列
        dic_old_pth = os.path.join(inputPath_obj, filename, filename + '_old_face_dict.pth')
        face_list_old = read_mesh_verts_idxs(dic_old_pth)
        print(f"{filename} 2.1 完成pth解析")
        # 2）标签新序列
        face_list_new, material_to_faces, mtl_file_line_num = read_obj_minimal(obj_file)  # 所有面索引
        face_list_new = np.array(face_list_new)
        print(f"{filename} 2.2 完成面片序列解析")
        # 建立新旧面片顺序的映射
        # (有点慢)
        new_old_idx = create_index_array(face_list_old, face_list_new)  # 索引,记录旧的映射到新的
        print(f"{filename} 2.3 完成新旧面索引映射")
        # 更新标签列表
        labels_new = dict()
        for key, face_list in class_labels.items():  # wigwam
            labels_new[key] =  new_old_idx[face_list]
        # 3) 更新打标签后的obj文件
        label_obj(labels_new, classlist, material_to_faces, obj_file, mtl_file_line_num)
        print(f"{filename} 2 完成obj打标签")
        # 3 点云分割
        # 1) mesh拆分
        print(f"{filename} 3 开始mesh点云化处理")
        meshes = mesh_fuc.crop_mesh_by_group(obj_file)
        print(f"{filename} 3.1 完成mesh分块")
        # 2) mesh点云化
        merged_cloud = np.empty((0, 7))
        for name,mesh in meshes.items():
            cloud_seg = mesh_fuc.mesh_2_points(mesh, subdense, min_point_Num=1)
            if len(cloud_seg.colors) <= 0:
                print(f"没有颜色，查看是否有贴图！")
            # 6 构造np格式点云(打标签单独封装功能)
            coord_seg = np.hstack((np.array(cloud_seg.points), np.array(cloud_seg.colors) * 255))
            lablei = label_list[classlist.index(name)]
            points_seg_np = np.hstack((coord_seg, np.full((coord_seg.shape[0], 1), lablei)))  # x,y,z, r,g,b, label
            # 7 分块点云合并
            merged_cloud = np.vstack((merged_cloud, points_seg_np))  # shape(N,7)
        # 8 写出点云
        write_ply_file(cloud_file, merged_cloud)
        print(f"{filename} 3 完成点云化")







