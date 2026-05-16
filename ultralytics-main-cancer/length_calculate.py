#
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
#
# def trace_ordered_skeleton(skeleton):
#     """追踪有序骨架点"""
#     # 寻找端点（只有一个邻居的点）
#     endpoints = []
#     skel = skeleton.astype(np.uint8)
#     for y in range(1, skel.shape[0] - 1):
#         for x in range(1, skel.shape[1] - 1):
#             if skel[y, x] == 0:
#                 continue
#             # 计算8邻域内白点数量
#             neighbors = np.sum(skel[y - 1:y + 2, x - 1:x + 2]) - 1
#             if neighbors == 1:
#                 endpoints.append((y, x))
#
#     # 处理特殊情况：环形结构无端点
#     if not endpoints:
#         ys, xs = np.where(skel)
#         return list(zip(ys, xs)) if len(ys) > 0 else []
#
#     # 从端点开始追踪路径
#     path = []
#     current = endpoints[0]
#     prev = (-1, -1)
#
#     while True:
#         path.append(current)
#         y, x = current
#
#         # 寻找下一个相邻点（8邻域）
#         next_points = []
#         for dy in [-1, 0, 1]:
#             for dx in [-1, 0, 1]:
#                 if dy == 0 and dx == 0:
#                     continue
#                 ny, nx = y + dy, x + dx
#                 if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
#                     if skel[ny, nx] and (ny, nx) != prev:
#                         next_points.append((ny, nx))
#
#         # 终止条件：到达端点或分叉点
#         if len(next_points) != 1:
#             break
#
#         prev = current
#         current = next_points[0]
#
#         # 防止无限循环
#         if current in path:
#             break
#
#     return path
#
#
# def calculate_crack_length(input_path):
#     """计算裂缝长度（积分法）并绘制骨架"""
#     # 读取图像并二值化
#     img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
#     # 骨架提取
#     skeleton = skeletonize(binary // 255)
#
#     # 获取有序骨架点
#     ordered_points = trace_ordered_skeleton(skeleton)
#
#     # 创建彩色图像用于绘制
#     color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#     # 绘制骨架
#     for point in ordered_points:
#         cv2.circle(color_img, point[::-1], 1, (0, 255, 0), -1)
#
#     # 计算折线积分长度并在图像上标注
#     total_length = 0.0
#     for i in range(1, len(ordered_points)):
#         y1, x1 = ordered_points[i - 1]
#         y2, x2 = ordered_points[i]
#         dx = x2 - x1
#         dy = y2 - y1
#         segment_length = np.hypot(dx, dy)  # 等效于sqrt(dx² + dy²)
#         total_length += segment_length
#         cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
#     # 在图像上标注总长度
#     cv2.putText(color_img, f"Crack Length: {total_length:.2f} pixels", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     # 显示结果图像
#     cv2.imshow('Crack Skeleton', color_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return total_length
#
#
# # 使用示例
# if __name__ == "__main__":
#     input_path = "segment_image/1_result.jpg"
#     crack_length = calculate_crack_length(input_path)
#     print(f"Crack Length: {crack_length:.2f} pixels")
#
#
#
import cv2
import numpy as np
from skimage.morphology import skeletonize


def trace_ordered_skeleton(skeleton):
    """追踪有序骨架点并返回路径"""
    # 寻找端点（只有一个邻居的点）
    endpoints = []
    skel = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # 使用卷积加速邻域计算
    conv = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((skel == 1) & (conv == 1))
    endpoints = list(zip(ys, xs))

    # 处理特殊情况：环形结构无端点
    if not endpoints:
        ys, xs = np.where(skel)
        return list(zip(ys, xs)) if len(ys) > 0 else []

    # 从端点开始追踪路径
    path = []
    current = endpoints[0]
    prev = (-1, -1)

    while True:
        path.append(current)
        y, x = current

        # 寻找下一个相邻点（8邻域）
        next_points = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx] and (ny, nx) != prev:
                        next_points.append((ny, nx))

        # 终止条件：到达端点或分叉点
        if len(next_points) != 1:
            break

        prev = current
        current = next_points[0]

        # 防止无限循环
        if current in path:
            break

    return path


def visualize_crack_analysis(input_path, output_path, scale_factor=0.1):
    """可视化裂缝分析过程"""
    # 读取图像并二值化
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 创建可视化图像
    vis_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 骨架提取
    skeleton = skeletonize(binary // 255)

    # 绘制原始骨架（红色）
    ys, xs = np.where(skeleton)
    vis_img[ys, xs] = (0, 0, 255)  # BGR格式红色

    # 获取有序骨架点
    ordered_points = trace_ordered_skeleton(skeleton)

    # 计算长度并绘制积分路径
    total_length = 0.0
    if len(ordered_points) >= 2:
        # 绘制端点（蓝色）
        y, x = ordered_points[0]
        cv2.circle(vis_img, (x, y), 5, (255, 0, 0), -1)
        y, x = ordered_points[-1]
        cv2.circle(vis_img, (x, y), 5, (255, 0, 0), -1)

        # 绘制积分路径（绿色）
        for i in range(1, len(ordered_points)):
            y1, x1 = ordered_points[i - 1]
            y2, x2 = ordered_points[i]
            dx = x2 - x1
            dy = y2 - y1
            segment_length = np.hypot(dx, dy)
            total_length += segment_length

            # 绘制线段和中间点
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis_img, (x2, y2), 2, (200, 200, 0), -1)

    # # 添加测量结果
    # text = f"Length: {total_length:.2f}px ({total_length * scale_factor:.2f}mm)"
    # cv2.putText(vis_img, text, (20, 40),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    #             lineType=cv2.LINE_AA)

    # 保存可视化结果
    cv2.imwrite(output_path, vis_img)
    return total_length


# 使用示例
if __name__ == "__main__":
    input_img = "segment_image/1016_close.jpg"
    output_img = "segment_image/analysis_result1016.jpg"

    # 假设每个像素对应0.1mm (需要根据实际标定修改)
    pixel_to_mm = 0.1

    length = visualize_crack_analysis(
        input_img,
        output_img,
        scale_factor=pixel_to_mm
    )

    print(f"裂缝像素长度: {length:.2f}")
    print(f"实际物理长度: {length * pixel_to_mm:.2f}mm")