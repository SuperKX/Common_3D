import open3d as o3d
import numpy as np

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



if __name__ == '__main__':

    # 1 测试读入输入数据裁剪并点云化
    if True:
        # 1 obj
        input_obj_path = r'E:\LabelScripts\testdata\wraptest\27DATA19\obj\seg15\seg15.obj'
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
        pth_fileP = r'E:\LabelScripts\testdata\wraptest\27DATA19\pth\seg15.pth'
        class_labels = read_dict_to_binary(pth_fileP)
        for key,face_list in class_labels.items():
            # face_list = [0,1,2,3]
            if len(face_list) == 0:
                continue
            print(f"标签: {key}，数量：{len(face_list)}, 最大值：{max(face_list)}，最小值：{min(face_list)}")
            mesh_seg = crop_mesh_with_face_ids(mesh, face_list, True)
            tes = 1
