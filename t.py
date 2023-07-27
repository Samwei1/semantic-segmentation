import numpy as np
from scipy.spatial import Delaunay

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def find_polygon_visual_center(polygon):
    # 使用Delaunay三角剖分将凹多边形转化为多个三角形
    tri = Delaunay(polygon)

    # 存储所有三角形的重心坐标
    triangle_centers = []

    # 计算所有三角形的重心
    for simplex in tri.simplices:
        vertices = polygon[simplex]
        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1])
        triangle_centers.append((center_x, center_y))

    # 计算所有三角形的重心的平均值，得到凹多边形的视觉中心
    center_x = np.mean([center[0] for center in triangle_centers])
    center_y = np.mean([center[1] for center in triangle_centers])
    
    visual_center = (center_x, center_y)

    # 判断视觉中心是否在凹多边形内部，如果不在，则向凹多边形内部移动，直到找到一个在内部的点
    while not point_in_polygon(visual_center, polygon):
        for i in range(len(polygon)):
            visual_center = ((visual_center[0] + polygon[i][0]) / 2, (visual_center[1] + polygon[i][1]) / 2)

    return visual_center

# Example usage:
polygon = [(0, 0), (4, 0), (4, 3), (2, 5), (0, 3)]  # Replace with your concave polygon's vertices
center_x, center_y = find_polygon_visual_center(polygon)
print(f"Concave Polygon Visual Center: ({center_x}, {center_y})")
