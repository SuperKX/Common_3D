import open3d as o3d


def visualize(mesh):
    o3d.visualization.draw_geometries([mesh])

# 可视化
def visualize2(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
