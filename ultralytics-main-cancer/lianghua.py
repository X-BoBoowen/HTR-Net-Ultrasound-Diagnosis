import cv2
import numpy as np

# 读取输入图像
image = cv2.imread(r'D:\waibao\yolo11\ultralytics-main\runs\segment\predict7\1016.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



# 创建调节窗口
cv2.namedWindow('Controls')
cv2.resizeWindow('Controls', 600, 300)

# 初始化滑动条参数
cv2.createTrackbar('H Min', 'Controls', 90, 179, lambda x: None)
cv2.createTrackbar('H Max', 'Controls', 130, 179, lambda x: None)
cv2.createTrackbar('S Min', 'Controls', 50, 255, lambda x: None)
cv2.createTrackbar('S Max', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('V Min', 'Controls', 50, 255, lambda x: None)
cv2.createTrackbar('V Max', 'Controls', 255, 255, lambda x: None)

# 形态学处理核
kernel = np.ones((7, 7), np.uint8)

# 全局变量存储选择的区域
drawing = False
start_point = (0, 0)
current_point = (0, 0)
selected_rect = None


# 鼠标回调函数
def select_region(event, x, y, flags, param):
    global drawing, start_point, current_point, selected_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)
        selected_rect = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)
        # 计算矩形区域
        x1 = max(0, min(start_point[0], current_point[0]))
        y1 = max(0, min(start_point[1], current_point[1]))
        x2 = min(image.shape[1], max(start_point[0], current_point[0]))
        y2 = min(image.shape[0], max(start_point[1], current_point[1]))
        selected_rect = (x1, y1, x2, y2)


# 绑定鼠标回调到结果窗口
cv2.namedWindow('Segmented Result', 0)
cv2.setMouseCallback('Segmented Result', select_region)

while True:
    # 获取滑动条当前值
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')

    # 定义HSV阈值范围
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # 创建颜色掩膜
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学处理
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.namedWindow("mask",0)
    cv2.imshow("mask",mask)
    # cv2.imwrite('D:/waibao/yolo11/ultralytics-main/segment_image/mask_close1.jpg', mask)
    # 创建结果图像（黑色背景）
    result = np.zeros_like(image)
    result[mask == 255] = (255, 255, 255)

    # 绘制当前选择的矩形区域
    if drawing or (start_point != (0, 0) and current_point != (0, 0)):
        cv2.rectangle(result, start_point, current_point, (0, 0, 0), 2)

    # 处理选中的区域
    if selected_rect is not None:
        x1, y1, x2, y2 = selected_rect
        # 提取原图和结果中的区域
        roi_original = image[y1:y2, x1:x2]
        roi_result = result[y1:y2, x1:x2]

        # 放大并显示原图区域
        if roi_original.size > 0:
            zoomed_original = cv2.resize(roi_original, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Zoomed Original', zoomed_original)
        # 放大并显示结果区域
        print(roi_result.size)
        if roi_result.size > 0:
            zoomed_result = cv2.resize(roi_result, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Zoomed Result', zoomed_result)
        cv2.imwrite('D:/waibao/yolo11/ultralytics-main/segment_image/zoomed_result1016.jpg', zoomed_result)
        print("image saved")
        #对筛选出来的区域画出

    # 显示实时结果
    cv2.imshow('Segmented Result', result)

    # 按q退出循环q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 保存最终结果（退出循环后保存）
cv2.imwrite('D:/waibao/yolo11/ultralytics-main/segment_image/1016_close.jpg', result)
cv2.destroyAllWindows()





# #canny轮廓检测
# import cv2
# import numpy as np
#
# # 读取输入图像
# image = cv2.imread('D:/waibao/yolo11/ultralytics-main/segment_image/segment1.jpg')
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#
# def canny_edge_detection(input_path, output_path, low_threshold=50, high_threshold=150):
#     # 读取图像
#     image = cv2.imread(input_path)
#     if image is None:
#         raise ValueError("Image not found or unable to load.")
#
#     # 转换为灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 应用高斯模糊
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#
#     # Canny边缘检测
#     edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
#
#     # # 显示结果
#     # cv2.imshow('Original Image', image)
#     # cv2.imshow('Edges Detected', edges)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # 保存结果
#     cv2.imwrite(output_path, edges)
# output_path = "segment_image/canny_edges4.jpg"
# input_path ="D:/waibao/yolo11/ultralytics-main/segment_image/segment4.jpg"
# canny_edge_detection(input_path, output_path, low_threshold=20, high_threshold=50)