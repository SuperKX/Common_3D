import open3d as o3d
import numpy as np
from PIL import Image
import os
import glob


# mesh 点云化
def mesh_2_points(mesh, subdense, min_point_Num=1):
    '''
    mesh点云化
    mesh 输入的o3d格式mesh
    subdense 采样密度（mesh面积上每subdense间隔采一个点）
    min_point_Num 最少采样点数量
    '''
    area = mesh.get_surface_area()
    number_points = int(area / (subdense * subdense)) #+ min_point_Num  # 至少一个点
    number_points = min_point_Num if number_points < min_point_Num else number_points
    # 点云化
    pcd = mesh.sample_points_uniformly(number_of_points=number_points)
    return pcd


def obj_merge_folder(obj_folder_input, obj_folder_output):
    '''
        合并当前文件夹中所有obj文件
        注意：
            1）文件名来自与 obj_folder_input 最后一级名称
        obj_folder_input: obj 文件夹路径
        obj_folder_output: obj 写出地址
    '''
    # 获取多级文件夹中所有obj文件
    import path.path_process as pathlib
    obj_files = pathlib.get_all_files_ext(obj_folder_input, ".obj")
    # obj_files = glob.glob(os.path.join(obj_folder_input, "*.obj"))
    file_name = os.path.basename(obj_folder_input)
    obj_merged_file = os.path.join(obj_folder_output, file_name + ".obj")

    if not obj_files:
        print("文件夹中没有找到obj文件")
        return False
    os.makedirs(obj_folder_output, exist_ok=True)

    merged_mesh = None
    # 逐个处理文件，这是最直接的方式
    for i, obj_file in enumerate(obj_files):
        try:
            # 读取obj文件（带纹理）
            mesh = o3d.io.read_triangle_mesh(obj_file, True)

            if merged_mesh is None:
                merged_mesh = mesh
            else:
                merged_mesh += mesh
            # 可选：手动删除临时mesh对象以提示垃圾回收
            del mesh

            if (i+1)%30 == 0:
                print(f"处理文件数量进度: {i + 1}/{len(obj_files)}")
            if i+1 == len(obj_files):
                print(f"处理文件数量进度: {i+1}/{i+1}")

        except Exception as e:
            print(f"警告：读取文件 {obj_file} 失败，跳过该文件。错误：{e}")
            continue

    if merged_mesh is None:
        print("没有成功读取任何obj文件")
        return False

    # 写出合并后的mesh
    try:
        o3d.io.write_triangle_mesh(obj_merged_file, merged_mesh)
        print(f"成功合并 {obj_files} 个obj文件到: {obj_merged_file}")
        return True
    except Exception as e:
        print(f"写出文件失败：{e}")
        return False


# def transform_mesh(input_obj, output_obj, transformation_matrix):
#     """
#     读取OBJ格式的mesh，应用矩阵变换，然后保存结果
#
#     参数:
#         input_obj (str): 输入OBJ文件路径
#         output_obj (str): 输出OBJ文件路径
#         transformation_matrix (np.ndarray): 4x4变换矩阵
#     """
#     # 读取mesh
#     mesh = o3d.io.read_triangle_mesh(input_obj)
#
#     # 检查是否成功读取mesh
#     if not mesh.has_vertices():
#         raise ValueError("无法读取mesh或mesh没有顶点")
#
#     # 检查变换矩阵是否为4x4
#     if transformation_matrix.shape != (4, 4):
#         raise ValueError("变换矩阵必须是4x4的numpy数组")
#
#     # 应用变换
#     mesh.transform(transformation_matrix)
#
#     # 写出mesh
#     os.makedirs(os.path.dirname(output_obj), exist_ok=True)
#     o3d.io.write_triangle_mesh(output_obj, mesh)
#
#     print(f"变换后的mesh已保存到: {output_obj}")



def seg_mesh_with_face(groupi,points,vertex_textures):
    '''
    分割一块mesh
    (待优化)：删除多余点及法向坐标
    '''
    # 1 处理数据
    textures = []
    new_uv_ids = []  # 面片贴图对应的土坯那编号
    new_triangles = []
    new_uvs_index =[]  # 面片uv的索引，后面需要映射到坐标
    uv_ids=0  # 当前uv贴图编号
    for mtl_name,mtl_info in groupi.items():
        # 1) 添加纹理图
        # 读取纹理图片
        texture_image = Image.open(mtl_info['pic'])
        texture_array = np.array(texture_image)
        # uv坐标跟图片坐标不一致！！
        texture_array = np.flipud(texture_array) # 垂直翻转图像数据
        texture_array = np.ascontiguousarray(texture_array)  # 内存内保持连续
        # 将纹理赋给 mesh
        textures.append(o3d.geometry.Image(texture_array))

        # 2) 添加面片
        new_triangles.extend(mtl_info['face'])
        # 3) 添加uv
        new_uvs_index.extend(mtl_info['uv'])
        # 4) uv图片索引
        new_uv_ids.extend([uv_ids]*len(mtl_info['uv']))
        uv_ids+=1

    # 2 uv索引转坐标
    new_uvs = [[vertex_textures[idx[0]], vertex_textures[idx[1]], vertex_textures[idx[2]]]for idx in new_uvs_index]
    new_uvs = np.array(new_uvs)  # 确保是 ndarray
    new_uvs = new_uvs.reshape(-1, 2)  # -1 表示自动计算 n*3
    # 3 构造mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    # 纹理需要以下3个信息
    new_mesh.triangle_uvs = o3d.utility.Vector2dVector(new_uvs)
    new_mesh.textures = textures
    new_mesh.triangle_material_ids = o3d.utility.IntVector(np.array(new_uv_ids))

    # 4 可视化（验证纹理是否正确）
    # o3d.visualization.draw_geometries([new_mesh])
    return new_mesh

def crop_mesh_by_group(obj_filename):
    '''
    读入 obj文件，根据group分割成多个mesh（o3d）
    '''
    '''
    groups={
        group1:{                # group名
            mtl1:{              # 纹理贴图名
                'face':[]       # 面片索引列表
                'uv':[]         # 面片贴图索引
                'pic'='str'     # 纹理图地址
                }
            }
        }
    '''
    groups = {}
    points = []
    vertex_textures = []
    current_group = ''
    # 纹理相关
    mtllib = ''  # mtl文件名 ’seg28866.mtl‘
    pic_path=dict()  #记录所有纹理名，对应的贴图地址
    # 返回值
    meshes = dict()  # 返回构造好的mesh
    with open(obj_filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                # Vertex coordinates
                vertex = [float(x) for x in line[2:].split()]
                points.append(vertex)
            elif line.startswith("vt "):
                # Vertex coordinates
                vt = [float(x) for x in line[3:].split()]
                vertex_textures.append(vt)
            elif line.startswith("g "):
                # Group name
                group_name = line[2:]
                groups[group_name] = dict()
                current_group = group_name  # 当前group
            elif line.startswith("usemtl "):
                mtl_name = line[7:]
                current_mtl = mtl_name  # 当前mtl
                if current_group == '':  # 暂时不处理没有打标签的mesh
                    continue
                groups[current_group][current_mtl] = {'face': [], 'uv': []}
            elif line.startswith("f "):
                # Face indices
                face = [int(x.split("/")[0]) - 1 for x in line[2:].split()]  # OBJ indices start from 1
                face_uv = [int(x.split("/")[1]) - 1 for x in line[2:].split()]  # uv坐标
                groups[current_group][current_mtl]['face'].append(face)
                groups[current_group][current_mtl]['uv'].append(face_uv)
            elif line.startswith("mtllib "):
                mtllib = line[7:]

    # 2 纹理图找地址
    if mtllib == '':
        raise ValueError("obj文件中没有找到mtllib 文件")
    mtllip_file = os.path.join(os.path.split(obj_filename)[0],mtllib)
    with open(mtllip_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("newmtl "):
                current_mtl_name = line[7:]
            elif 'map_Kd ' in line:
                filename = line.split()[-1]
                pic_path[current_mtl_name] = os.path.join(os.path.split(obj_filename)[0],filename)
    # 纹理地址加到groups中：
    for class_name,groupi in groups.items():
        for mtl_name, mtl in groupi.items():
            mtl['pic'] = pic_path[mtl_name]

    # 3 逐个分割
    for class_name,groupi in groups.items():
        meshi = seg_mesh_with_face(groupi, points, vertex_textures)
        meshes[class_name] = meshi
        print(f"    {class_name} 分割完成")

    return meshes

# mesh 裁剪
def crop_mesh_with_face_ids(mesh, face_list, bool_visualize=False):
    """
    通过面片索引列表，拆分obj文件，
    mesh  # 带纹理的o3d格式mesh  = o3d.io.read_triangle_mesh(input_obj_path, True)
    face_list 目标面片的索引列表 list
    return 裁剪后mesh
    注意：
        1）目前仅支持带纹理的，没做兼容
        2）目前写出还有问题，需要排查
    """
    try:
        # 0 判断合法性
        if not mesh.has_triangles() or not mesh.has_vertices():
            raise ValueError("错误：网格为空或无效。")

        # 1 裁剪
        # 1.0 构造新旧映射-顶点
        # o3d.visualization.draw_geometries([mesh])
        triangles = np.asarray(mesh.triangles)
        new_triangles_old_id = triangles[face_list]   # 老的面片id

        if False: # 测试：只看裁剪位置是否正确
            mesh.triangles = o3d.utility.Vector3iVector(new_triangles_old_id)
            o3d.visualization.draw_geometries([mesh])

        unique_vertex_indices = np.unique(new_triangles_old_id)  # 分割后mesh用到的顶点-旧编号
        vertex_index_mapping = {old_index: new_index for new_index, old_index in enumerate(unique_vertex_indices)}  # 顶点的新旧序号映射
        # 1.1 segmesh面片顶点-新索引 new_triangles
        new_triangles = np.array([[vertex_index_mapping[v] for v in triangle] for triangle in new_triangles_old_id])  # 更新面顶点索引
        # 1.2 新顶点坐标
        # 提取与选定面片对应的顶点、纹理坐标和顶点颜色。
        vertices = np.asarray(mesh.vertices)
        new_vertices = vertices[unique_vertex_indices]
        # 1.3 uv贴图更新
        uvs = np.asarray(mesh.triangle_uvs)  # Vector2dVector（3*nf,2)
        uvs_list = [i*3+j for i in face_list for j in range(3)]  # 裁剪的面片列表，找到uv中对应面片位置，更新进来
        new_uvs = uvs[uvs_list]
        # 1.4 uv图片索引
        uv_ids = np.asarray(mesh.triangle_material_ids)
        new_uv_ids = uv_ids[face_list]

        # 2 构造mesh
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        # 纹理需要以下3个信息
        new_mesh.triangle_uvs = o3d.utility.Vector2dVector(new_uvs)
        new_mesh.textures = mesh.textures
        new_mesh.triangle_material_ids = o3d.utility.IntVector(new_uv_ids)

        # 可视化
        if bool_visualize:
            o3d.visualization.draw_geometries([new_mesh])

        return new_mesh

    except Exception as e:
        print(f"裁剪和保存网格时发生错误：{e}")
        return False

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


import re


def process_vertex_line(line):
    """
    处理OBJ文件中的顶点坐标行，将z坐标移动到最前面
    参数:
    line (str): OBJ文件中的顶点行，格式如 "v  2077.9175 243.6621 3411.6072"
    返回:
    str: 处理后的顶点行，z坐标在最前面
    """
    # 去除前后空格并按空格分割
    parts = line.strip().split()

    # 检查是否为顶点行
    if len(parts) < 4 or parts[0] != 'v':
        return line  # 不是顶点行，原样返回

    # 提取坐标 (v, x, y, z)
    v_flag = parts[0]
    x = parts[1]
    y = parts[2]
    z = parts[3]

    # 重新排列为 (v, z, x, y)
    return f"{v_flag}  {z} {x} {y}"

def split_obj_meshes(obj_file_path, output_folder, process_Trans=True):
    """
    分割OBJ格式的网格文件，将每个独立的mesh保存为单独的文件，
    保留原始文件中的MTL材质引用，并修复顶点索引以适应局部坐标系
    注意：
        1）此处分割默认再obj文件中相互独立。比如每个mesh为恋曲区间的v、vt、vn、f数据。
    参数:
        obj_file_path (str): OBJ文件的完整路径
        output_folder (str): 输出文件夹路径，用于保存分割后的文件
        process_Trans: 处理变换
    返回:
        str: 输出文件夹的路径，如果发生错误则返回None
    """
    try:
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 读取OBJ文件内容
        with open(obj_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 提取所有MTL文件引用行
        mtl_lines = []
        mtl_pattern = re.compile(r'^mtllib\s+.+?\.mtl$', re.MULTILINE)
        for match in mtl_pattern.finditer(content):
            mtl_lines.append(match.group(0))

        # 正则表达式匹配每个mesh的开头: #\n# object 名称\n#\n
        mesh_pattern = re.compile(r'#\s*\n#\s*object\s+(.+?)\s*\n#\s*\n', re.DOTALL)
        matches = list(mesh_pattern.finditer(content))

        if not matches:
            print("未找到任何符合格式的mesh，可能文件格式不符合要求")
            return None

        # 首先解析整个文件，记录所有顶点、法向量和纹理坐标
        all_vertices = []  # 存储所有v行
        all_normals = []  # 存储所有vn行
        all_texcoords = []  # 存储所有vt行

        # 匹配所有顶点数据行
        vertex_pattern = re.compile(r'^(v|vn|vt)\s+.*$', re.MULTILINE)
        for line in vertex_pattern.finditer(content):
            line_content = line.group(0)
            if line_content.startswith('v '):
                if process_Trans:
                    processed_line = process_vertex_line(line_content)
                all_vertices.append(processed_line)
            elif line_content.startswith('vn '):
                all_normals.append(line_content)
            elif line_content.startswith('vt '):
                all_texcoords.append(line_content)

        # 获取顶点数据的数量，用于边界检查
        num_vertices = len(all_vertices)
        num_normals = len(all_normals)
        num_texcoords = len(all_texcoords)

        # 分割并保存每个mesh
        for i, match in enumerate(matches):
            # 提取mesh名称
            mesh_name = match.group(1).strip()
            print(f"处理 mesh: {mesh_name}")

            # 确定当前mesh的内容范围
            start_index = match.start()
            if i < len(matches) - 1:
                end_index = matches[i + 1].start()
            else:
                end_index = len(content)  # 最后一个mesh到文件结尾

            # 提取当前mesh的完整内容
            mesh_content = content[start_index:end_index]

            # 收集当前mesh使用的所有索引
            used_v_indices = set()  # 使用的顶点索引
            used_vt_indices = set()  # 使用的纹理坐标索引
            used_vn_indices = set()  # 使用的法向量索引

            # 匹配所有面的定义
            face_pattern = re.compile(r'f\s+((?:\S+\s+)+)', re.DOTALL)
            for face in face_pattern.finditer(mesh_content):
                face_data = face.group(1)
                # 分割面的各个顶点引用
                vertices_in_face = re.findall(r'(\d+/?\d*/?\d*)', face_data)

                for vert in vertices_in_face:
                    if not vert:
                        continue

                    # 解析顶点索引格式，可能是: v1, v1/vt1, v1/vt1/vn1, v1//vn1
                    parts = vert.split('/')

                    # 处理顶点索引 (支持负数索引)
                    if parts[0]:
                        try:
                            v_idx = int(parts[0])
                            # 处理负数索引（OBJ格式支持，从末尾开始计数）
                            if v_idx < 0:
                                v_idx = num_vertices + v_idx + 1  # 转换为正数索引

                            # 检查索引是否有效
                            if 1 <= v_idx <= num_vertices:
                                used_v_indices.add(v_idx)
                            else:
                                print(f"警告: 顶点索引 {v_idx} 超出范围 (1-{num_vertices})，已跳过")
                        except ValueError:
                            print(f"警告: 无效的顶点索引 '{parts[0]}'，已跳过")

                    # 处理纹理坐标索引
                    if len(parts) > 1 and parts[1]:
                        try:
                            vt_idx = int(parts[1])
                            if vt_idx < 0:
                                vt_idx = num_texcoords + vt_idx + 1

                            if 1 <= vt_idx <= num_texcoords or num_texcoords == 0:
                                used_vt_indices.add(vt_idx)
                            else:
                                print(f"警告: 纹理坐标索引 {vt_idx} 超出范围 (1-{num_texcoords})，已跳过")
                        except ValueError:
                            print(f"警告: 无效的纹理坐标索引 '{parts[1]}'，已跳过")

                    # 处理法向量索引
                    if len(parts) > 2 and parts[2]:
                        try:
                            vn_idx = int(parts[2])
                            if vn_idx < 0:
                                vn_idx = num_normals + vn_idx + 1

                            if 1 <= vn_idx <= num_normals or num_normals == 0:
                                used_vn_indices.add(vn_idx)
                            else:
                                print(f"警告: 法向量索引 {vn_idx} 超出范围 (1-{num_normals})，已跳过")
                        except ValueError:
                            print(f"警告: 无效的法向量索引 '{parts[2]}'，已跳过")

            # 检查是否有可用顶点
            if not used_v_indices:
                print(f"警告: mesh '{mesh_name}' 没有有效的顶点索引，已跳过")
                continue

            # 创建全局索引到本地索引的映射
            sorted_v_indices = sorted(used_v_indices)
            sorted_vt_indices = sorted(used_vt_indices)
            sorted_vn_indices = sorted(used_vn_indices)

            v_map = {idx: i + 1 for i, idx in enumerate(sorted_v_indices)}
            vt_map = {idx: i + 1 for i, idx in enumerate(sorted_vt_indices)}
            vn_map = {idx: i + 1 for i, idx in enumerate(sorted_vn_indices)}

            # 提取当前mesh使用的顶点数据（添加边界检查）
            local_vertices = []
            for idx in sorted_v_indices:
                try:
                    local_vertices.append(all_vertices[idx - 1])
                except IndexError:
                    print(f"警告: 顶点索引 {idx} 超出顶点列表范围，已跳过")

            local_normals = []
            for idx in sorted_vn_indices:
                try:
                    local_normals.append(all_normals[idx - 1])
                except IndexError:
                    print(f"警告: 法向量索引 {idx} 超出法向量列表范围，已跳过")

            local_texcoords = []
            for idx in sorted_vt_indices:
                try:
                    local_texcoords.append(all_texcoords[idx - 1])
                except IndexError:
                    print(f"警告: 纹理坐标索引 {idx} 超出纹理坐标列表范围，已跳过")

            # 替换面中的索引为本地索引
            def replace_indices(match):
                prefix = match.group(1)
                indices = match.group(2)

                def replace_single_index(vert_match):
                    vert = vert_match.group(0)
                    parts = vert.split('/')
                    new_parts = []

                    # 处理顶点索引
                    if parts[0]:
                        try:
                            v_idx = int(parts[0])
                            # 处理负数索引
                            if v_idx < 0:
                                v_idx = num_vertices + v_idx + 1

                            if v_idx in v_map:
                                new_parts.append(str(v_map[v_idx]))
                            else:
                                new_parts.append('')
                                print(f"警告: 无法映射顶点索引 {parts[0]}")
                        except (KeyError, ValueError):
                            new_parts.append('')
                    else:
                        new_parts.append('')

                    # 处理纹理坐标索引
                    if len(parts) > 1:
                        if parts[1]:
                            try:
                                vt_idx = int(parts[1])
                                if vt_idx < 0:
                                    vt_idx = num_texcoords + vt_idx + 1

                                if vt_idx in vt_map:
                                    new_parts.append(str(vt_map[vt_idx]))
                                else:
                                    new_parts.append('')
                                    print(f"警告: 无法映射纹理坐标索引 {parts[1]}")
                            except (KeyError, ValueError):
                                new_parts.append('')
                        else:
                            new_parts.append('')

                    # 处理法向量索引
                    if len(parts) > 2:
                        if parts[2]:
                            try:
                                vn_idx = int(parts[2])
                                if vn_idx < 0:
                                    vn_idx = num_normals + vn_idx + 1

                                if vn_idx in vn_map:
                                    new_parts.append(str(vn_map[vn_idx]))
                                else:
                                    new_parts.append('')
                                    print(f"警告: 无法映射法向量索引 {parts[2]}")
                            except (KeyError, ValueError):
                                new_parts.append('')
                        else:
                            new_parts.append('')

                    return '/'.join(new_parts).rstrip('/')

                updated_indices = re.sub(r'\d+/?\d*/?\d*', replace_single_index, indices)
                return prefix + updated_indices

            # 替换所有面的索引
            updated_mesh_content = re.sub(r'(f\s+)(.*?)(?=\n|$)', replace_indices, mesh_content, flags=re.DOTALL)

            # 构建新的mesh内容
            new_content_parts = []

            # 添加MTL引用
            if mtl_lines:
                new_content_parts.extend(mtl_lines)
                new_content_parts.append('')

            # 添加顶点数据
            new_content_parts.extend(local_vertices)
            if local_texcoords:
                new_content_parts.extend(local_texcoords)
            if local_normals:
                new_content_parts.extend(local_normals)
            new_content_parts.append('')

            # 添加处理后的mesh内容（不含原始顶点数据）
            filtered_lines = []
            for line in updated_mesh_content.split('\n'):
                stripped_line = line.strip()
                if not stripped_line.startswith(('v ', 'vn ', 'vt ')):
                    filtered_lines.append(line)

            new_content_parts.extend(filtered_lines)

            # 合并所有部分
            final_content = '\n'.join(new_content_parts)

            # 处理文件名
            safe_name = re.sub(r'[^\w\s-]', '', mesh_name).strip().replace(' ', '_')
            if not safe_name:
                safe_name = f"mesh_{i + 1}"

            # 构建输出文件路径
            output_file_path = os.path.join(output_folder, f"{safe_name}.obj")

            # 写入文件
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(final_content)

            print(
                f"已保存: {output_file_path} (顶点: {len(local_vertices)}, 纹理坐标: {len(local_texcoords)}, 法向量: {len(local_normals)})")

        print(f"\n所有mesh已成功分割，共分割出 {len(matches)} 个文件")
        if mtl_lines:
            print(f"已保留MTL引用: {', '.join(mtl_lines)}")
        return output_folder

    except FileNotFoundError:
        print(f"错误: 找不到文件 {obj_file_path}")
    except PermissionError:
        print(f"错误: 没有权限访问文件或文件夹，请检查权限设置")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        # 添加更详细的错误信息，帮助调试
        import traceback
        print("错误详情:")
        traceback.print_exc()

    return None


def rotate_obj_around_y_axis(input_path, output_path, rotation_center, rotation_angle_deg):
    """
    旋转OBJ文件绕Y轴

    参数:
        input_path: 输入OBJ文件路径
        output_path: 输出OBJ文件路径
        rotation_center: 旋转中心点，格式为(x, y, z)
        rotation_angle_deg: 旋转角度（度）
    """
    import math
    # 确保旋转中心是NumPy数组
    rotation_center = np.array(rotation_center)

    # 转换为弧度
    rotation_angle_rad = math.radians(rotation_angle_deg)

    # 计算三角函数值
    cos_theta = math.cos(rotation_angle_rad)
    sin_theta = math.sin(rotation_angle_rad)

    # 创建绕Y轴的旋转矩阵
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    try:
        # 读取OBJ文件
        mesh = o3d.io.read_triangle_mesh(input_path, True)

        # 获取顶点数组
        vertices = np.asarray(mesh.vertices)

        # 1. 平移顶点到旋转中心
        vertices_centered = vertices - rotation_center

        # 2. 应用旋转矩阵
        vertices_rotated = np.dot(vertices_centered, rotation_matrix.T)

        # 3. 平移回原坐标系
        vertices_final = vertices_rotated + rotation_center

        # 更新网格顶点
        mesh.vertices = o3d.utility.Vector3dVector(vertices_final)

        # 处理法线（如果存在）
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals)
            normals_rotated = np.dot(normals, rotation_matrix.T)
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals_rotated)

        # 保存旋转后的OBJ
        # 创建输出文件夹
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"成功旋转并保存: {input_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"处理文件时出错: {input_path}")
        print(f"错误信息: {str(e)}")
        return False

if __name__ == '__main__':
    # 批量obj坐标变换
    if False:
        folders = r'J:\data\data_3d_backup\singlebuilding\yulongzhuangyuan\obj'
        folders_out =r'J:\data\data_3d_backup\singlebuilding\yulongzhuangyuan\obj_'
        os.listdir(folders)
        for filename in os.listdir(folders):
            file_path = os.path.join(folders, filename)
            input_obj =os.path.join(file_path, filename+".obj")
            output_obj = os.path.join(folders_out,filename, filename + ".obj")

            # 旋转中心
            center = (4.55765, -61.7772, 138.992)
            # TODO:旋转轴：当前写死y轴
            # 旋转角度
            angle = 2.4851
            # 单个文件旋转
            rotate_obj_around_y_axis(input_obj, output_obj, center, angle)

    # 0 mesh点云化
    # TODO: 未验证。
    if False:
        input_obj_path = r'D:\1temperorytest\20250915春棣-大厦点云化\-1F装修 - 三维视图 - {3D}.obj'
        mesh = o3d.io.read_triangle_mesh(input_obj_path, True)
        subdense = 0.0001
        pcd = mesh_2_points(mesh, subdense, min_point_Num=1)
        import points.points_io as pts_io

        file_path = r'D:\1temperorytest\20250915春棣-大厦点云化\-1F装修 - 三维视图 - {3D}.ply'
        pts_io.write_ply_file(file_path,pcd)

    # 1 测试读入输入数据裁剪并点云化
    if False:
        # 1 obj
        input_obj_path = r'E:\LabelScripts\testdata\wraptest\27DATA19\obj\seg28866\seg28866.obj'
        mesh = o3d.io.read_triangle_mesh(input_obj_path, True)
        # 2 label
        def read_dict_to_binary(filename):
            '''
            解析pth格式的字典，
            注意：
                1）wrp中标签下标从1开始，此处-1，跟面片下标对齐。
                2）默认每个标签的面片先排序。
            '''
            import pickle
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
        pth_fileP = r'E:\LabelScripts\testdata\wraptest\27DATA19\pth\seg28866.pth'
        class_labels = read_dict_to_binary(pth_fileP)

        # # tests
        # pth_fileP222 = r'E:\LabelScripts\testdata\wraptest\27DATA19\obj\seg28866\seg28866_old_face_dict.pth'
        # class_labels = read_dict_to_binary(pth_fileP222)

        for key,face_list in class_labels.items():
            # face_list = [0,1,2,3]
            if len(face_list) == 0:
                continue
            print(f"标签: {key}，数量：{len(face_list)}, 最大值：{max(face_list)}，最小值：{min(face_list)}")
            mesh_seg = crop_mesh_with_face_ids(mesh, face_list, True)
            tes = 1

    # 2 测试obj分割
    if False:
        # 示例：分割input.obj文件，保存到output_meshes文件夹
        input_obj_path = r"H:\1 业务功能\2LOD3\资料备份\【墙面语义重构】25Q3\2025.08.28西安数据验证\segobjs/经开区建筑.obj"  # 替换为实际的OBJ文件路径
        output_dir = r"H:\TestData\LOD3/output_seg2"  # 替换为实际的输出文件夹路径

        result = split_obj_meshes(input_obj_path, output_dir)
        if result:
            print(f"分割后的文件保存在: {result}")

    # 3 测试obj合并功能：
    if True:
        obj_folder_input = r"J:\TempProcess\singleBuilding0924\Suzhou\5postprocess\output\realmodel"
        obj_folder_output = r"J:\TempProcess\singleBuilding0924\Suzhou\5postprocess\output\realmodel_merge"

        # 批量合并多组obj
        # 遍历文件夹
        for filename in os.listdir(obj_folder_input):
            file_path = os.path.join(obj_folder_input, filename)
            file_out_path = os.path.join(obj_folder_output, filename)
            obj_merge_folder(file_path, file_out_path)