
import cv2
# import numpy as np
# from skimage.morphology import skeletonize
#
#
# def draw_crack_lines(input_path, output_path, window_size=10, spacing=10):
#     # 读取图像并二值化
#     binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
#
#     # 骨架提取
#     skeleton = skeletonize(binary // 255)
#
#     # 距离变换计算宽度
#     dist_img = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#
#     # 获取有序骨架点（简单直线近似）
#     skeleton_points = np.argwhere(skeleton)
#
#     # 按坐标排序获得近似路径（适用于较直裂缝）
#     skeleton_points = sorted(skeleton_points, key=lambda x: (x[0], x[1]))
#
#     # 准备输出图像
#     output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#
#     # 等距采样控制
#     last_point = None
#     half_window = window_size // 2
#
#     for y, x in skeleton_points:
#         # 间距控制
#         if last_point is not None:
#             dx = x - last_point[0]
#             dy = y - last_point[1]
#             if np.sqrt(dx ** 2 + dy ** 2) < spacing:
#                 continue
#         last_point = (x, y)
#
#         # 提取邻域内的骨架点
#         y_min = max(0, y - half_window)
#         y_max = min(binary.shape[0], y + half_window + 1)
#         x_min = max(0, x - half_window)
#         x_max = min(binary.shape[1], x + half_window + 1)
#
#         region = skeleton[y_min:y_max, x_min:x_max]
#         yy, xx = np.where(region)
#         xx += x_min
#         yy += y_min
#         points = np.column_stack((xx, yy))
#
#         if len(points) < 2:
#             continue
#
#         # PCA计算主方向
#         mean = np.mean(points, axis=0)
#         centered = points - mean
#         cov = np.cov(centered.T)
#         eigenvalues, eigenvectors = np.linalg.eigh(cov)
#         idx = eigenvalues.argsort()[::-1]
#         main_dir = eigenvectors[:, idx[0]]
#
#         # 计算法线方向
#         angle = np.arctan2(main_dir[1], main_dir[0])
#         perp_angle = angle + np.pi / 2
#
#         # 获取宽度
#         width = 2 * dist_img[y, x]
#
#         # 计算线段端点
#         dx = np.cos(perp_angle) * width / 2
#         dy = np.sin(perp_angle) * width / 2
#         pt1 = (int(round(x - dx)), int(round(y - dy)))
#         pt2 = (int(round(x + dx)), int(round(y + dy)))
#
#         # 绘制线段
#         cv2.line(output, pt1, pt2, (0, 0, 255), 1)
#
#     cv2.imwrite(output_path, output)
#
#
# # draw_crack_lines("input.png", "output.png", window_size=5, spacing=20)
# draw_crack_lines("segment_image/1016_close.jpg", "segment_image/output1016.jpg", window_size=800,spacing=100)


#
#
#
#


import cv2
import numpy as np
from skimage.morphology import skeletonize

def draw_crack_lines(input_path, output_path, window_size=10, spacing=10):
    # 读取图像并二值化
    binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # 骨架提取
    skeleton = skeletonize(binary // 255)

    # 距离变换计算宽度
    dist_img = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 获取有序骨架点
    skeleton_points = np.argwhere(skeleton)
    skeleton_points = sorted(skeleton_points, key=lambda x: (x[0], x[1]))

    # 准备输出图像和长度存储
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    lengths = []  # 存储所有线段长度
    half_window = window_size // 2
    last_point = None

    for y, x in skeleton_points:
        # 等距采样控制
        if last_point is not None:
            dx = x - last_point[0]
            dy = y - last_point[1]
            if np.sqrt(dx**2 + dy**2) < spacing:
                continue
        last_point = (x, y)

        # 邻域骨架点提取
        y_min = max(0, y - half_window)
        y_max = min(binary.shape[0], y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(binary.shape[1], x + half_window + 1)

        region = skeleton[y_min:y_max, x_min:x_max]
        yy, xx = np.where(region)
        xx += x_min
        yy += y_min
        points = np.column_stack((xx, yy))

        if len(points) < 2:
            continue

        # PCA方向计算
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        main_dir = eigenvectors[:, np.argmax(eigenvalues)]

        # 计算法线方向
        angle = np.arctan2(main_dir[1], main_dir[0])
        perp_angle = angle + np.pi/2

        # 获取宽度并计算线段端点
        width = 2 * dist_img[y, x]
        dx = np.cos(perp_angle) * width/2
        dy = np.sin(perp_angle) * width/2
        pt1 = (int(round(x - dx)), int(round(y - dy)))
        pt2 = (int(round(x + dx)), int(round(y + dy)))

        # 边界检查并绘制
        if all([0 <= pt1[0] < binary.shape[1], 0 <= pt1[1] < binary.shape[0],
                0 <= pt2[0] < binary.shape[1], 0 <= pt2[1] < binary.shape[0]]):
            # 计算线段长度（欧氏距离）
            line_length = np.linalg.norm(np.array(pt2) - np.array(pt1))
            if (line_length < 90):
                continue
            lengths.append(line_length)

            cv2.line(output, pt1, pt2, (0, 0, 255), 1)

    # 输出统计结果
    if lengths:
        avg_length = np.mean(lengths)
        print(f"线段数量: {len(lengths)}")
        print(f"平均长度: {avg_length:.2f} 像素")
        print(f"最大长度: {np.max(lengths):.2f} 像素")
        print(f"最小长度: {np.min(lengths):.2f} 像素")
    else:
        print("未检测到有效线段")

    cv2.imwrite(output_path, output)

# 使用示例（参数根据实际需求调整）
draw_crack_lines("segment_image/zoomed_result1016.jpg",
                "segment_image/output1016.jpg",
                window_size=500,
                spacing=100)