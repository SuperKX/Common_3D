import open3d as o3d
import numpy as np
from PIL import Image
import os

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


if __name__ == '__main__':

    # 1 测试读入输入数据裁剪并点云化
    if True:
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
