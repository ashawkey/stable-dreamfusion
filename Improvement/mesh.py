import pymesh

# 读取两个 mesh
filename1 = "mesh1.obj"
filename2 = "mesh2.obj"
mesh1 = pymesh.load_mesh(filename1)
mesh2 = pymesh.load_mesh(filename2)

# 获取两个 mesh 的顶点数轮廓范围 bbox_min 和 bbox_max
bbox_min, bbox_max = mesh1.bbox
bbox_min2, bbox_max2 = mesh2.bbox

# 计算两个 mesh 形状半径
radius = max(mesh1.bbox.norm(), mesh2.bbox.norm())

# 计算两个 mesh 的中心
center1 = 0.5 * (bbox_min + bbox_max)
center2 = 0.5 * (bbox_min2 + bbox_max2)

# 计算将 mesh1 放在原点的变换矩阵
T1 = -center1

# 计算将 mesh2 放到 mesh1 旁边的变换矩阵
T2 = -center2 + [radius, 0, 0]

# 应用变换矩阵到 mesh1 和 mesh2
mesh1.apply_transform(T1)
mesh2.apply_transform(T2)

# 合并两个 mesh
merged_mesh = pymesh.merge_meshes([mesh1, mesh2])

# 保存合并后的 mesh
pymesh.save_mesh("merged_mesh.obj", merged_mesh)
