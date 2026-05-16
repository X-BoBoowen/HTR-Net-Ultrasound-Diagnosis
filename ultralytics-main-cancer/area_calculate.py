import cv2
import numpy as np

def calculate_crack_area(input_path):
    # 读取图像并二值化
    binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化总面积
    total_area = 0

    # 计算每个轮廓的面积并累加
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area < 5000):
            continue
        print(area)

        total_area += area

    return total_area

# 使用示例
input_path = "segment_image/4_close.jpg"
crack_area = calculate_crack_area(input_path)
print(f"裂缝区域的白色面积: {crack_area} 像素")



