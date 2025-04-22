import open3d as o3d
import numpy as np

def crop_mesh_with_face_ids(mesh, face_list):
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
        triangles = np.asarray(mesh.triangles)
        new_triangles_old_id = triangles[face_list]   # 老的面片id
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
        new_mesh.triangle_uvs = o3d.utility.Vector2dVector(new_uvs)
        new_mesh.textures = mesh.textures
        new_mesh.triangle_material_ids = o3d.utility.IntVector(new_uv_ids)

        # 可视化
        # o3d.visualization.draw_geometries([new_mesh])

        return new_mesh

    except Exception as e:
        print(f"裁剪和保存网格时发生错误：{e}")
        return False



if __name__ == '__main__':

    # 1 测试读入输入数据裁剪并点云化
    if True:
        # 2 obj-seg
        input_obj_path = r'E:\LabelScripts\testdata\wraptest\out\wraptest\wraptest.obj'
        mesh = o3d.io.read_triangle_mesh(input_obj_path, True)
        face_list = [0,1,2,3]
        mesh_seg = crop_mesh_with_face_ids(mesh, face_list)


