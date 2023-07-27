import numpy as np
from scipy.spatial import Delaunay

def find_polygon_visual_center(polygon):
    # 将多边形转换为numpy数组
    polygon_points = np.array(polygon)

    # 执行多边形的三角剖分
    triangulation = Delaunay(polygon_points)

    # 计算三角形的视觉中心并累加
    center_x_sum = 0
    center_y_sum = 0
    num_triangles = triangulation.simplices.shape[0]

    for simplex in triangulation.simplices:
        triangle_points = polygon_points[simplex]
        triangle_center_x = np.mean(triangle_points[:, 0])
        triangle_center_y = np.mean(triangle_points[:, 1])
        center_x_sum += triangle_center_x
        center_y_sum += triangle_center_y

    # 计算多边形的视觉中心（三角形视觉中心的平均值）
    center_x = center_x_sum / num_triangles
    center_y = center_y_sum / num_triangles

    return center_x, center_y

# Example usage:
polygon = [(0, 0), (4, 0), (4, 3), (2, 5), (0, 3)]  # Replace with your concave polygon's vertices
center_x, center_y = find_polygon_visual_center(polygon)
print(f"Concave Polygon Visual Center: ({center_x}, {center_y})")
