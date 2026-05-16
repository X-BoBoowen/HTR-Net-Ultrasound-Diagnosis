# import cv2
# import numpy as np
# import os
# import glob
#
# # 输入输出路径设置
# predict_dir = r'D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-04\123'  # 预测图片目录
# original_dir = r'D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-04\qie'  # 原始图片目录
# output_dir = r'D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-04\\output'  # 输出目录
#
# # 创建输出目录
# os.makedirs(output_dir, exist_ok=True)
#
# # 初始化调节窗口（只创建一次）
# cv2.namedWindow('HSV Controls')
# cv2.resizeWindow('HSV Controls', 600, 300)
#
# # 创建滑动条（只创建一次）
# cv2.createTrackbar('H Min', 'HSV Controls', 100, 179, lambda x: None)
# cv2.createTrackbar('H Max', 'HSV Controls', 140, 179, lambda x: None)
# cv2.createTrackbar('S Min', 'HSV Controls', 50, 255, lambda x: None)
# cv2.createTrackbar('S Max', 'HSV Controls', 255, 255, lambda x: None)
# cv2.createTrackbar('V Min', 'HSV Controls', 50, 255, lambda x: None)
# cv2.createTrackbar('V Max', 'HSV Controls', 255, 255, lambda x: None)
#
# # 形态学处理核
# kernel = np.ones((7, 7), np.uint8)
#
# # 获取预测目录中的所有jpg文件
# predict_images = glob.glob(os.path.join(predict_dir, '*.jpg'))
#
# for predict_path in predict_images:
#     # 构建对应原始图片路径
#     filename = os.path.basename(predict_path)
#     original_path = os.path.join(original_dir, filename)
#
#     if not os.path.exists(original_path):
#         print(f"跳过未找到的原始图片: {original_path}")
#         continue
#
#     # 读取图片对
#     image = cv2.imread(predict_path)
#     image2 = cv2.imread(original_path)
#
#     if image is None or image2 is None:
#         print(f"图片读取失败: {predict_path} 或 {original_path}")
#         continue
#
#     # 统一尺寸
#     image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
#
#     # 转换到HSV色彩空间
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     # 重置滑动条到默认值
#     cv2.setTrackbarPos('H Min', 'HSV Controls', 100)
#     cv2.setTrackbarPos('H Max', 'HSV Controls', 140)
#     cv2.setTrackbarPos('S Min', 'HSV Controls', 50)
#     cv2.setTrackbarPos('S Max', 'HSV Controls', 255)
#     cv2.setTrackbarPos('V Min', 'HSV Controls', 50)
#     cv2.setTrackbarPos('V Max', 'HSV Controls', 255)
#
#     # 当前图片处理循环
#     while True:
#         # 获取滑动条参数
#         h_min = cv2.getTrackbarPos('H Min', 'HSV Controls')
#         h_max = cv2.getTrackbarPos('H Max', 'HSV Controls')
#         s_min = cv2.getTrackbarPos('S Min', 'HSV Controls')
#         s_max = cv2.getTrackbarPos('S Max', 'HSV Controls')
#         v_min = cv2.getTrackbarPos('V Min', 'HSV Controls')
#         v_max = cv2.getTrackbarPos('V Max', 'HSV Controls')
#
#         # 生成掩膜
#         lower = np.array([h_min, s_min, v_min])
#         upper = np.array([h_max, s_max, v_max])
#         mask = cv2.inRange(hsv, lower, upper)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#         # 应用掩膜
#         segmented = cv2.bitwise_and(image2, image2, mask=mask)
#
#         # 显示结果
#         cv2.imshow('Original Mask', mask)
#         cv2.imshow('Segmented Result', segmented)
#
#         # 等待按键
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):  # 保存并处理下一张
#             output_path = os.path.join(output_dir, filename)
#             cv2.imwrite(output_path, segmented)
#             print(f"已保存处理结果: {output_path}")
#             break
#         elif key == 27:  # ESC键退出程序
#             cv2.destroyAllWindows()
#             exit()
#
# # 最后关闭所有窗口
# cv2.destroyAllWindows()




import cv2
import numpy as np
import os
import glob

# 输入输出路径设置
predict_dir = r'D:\waibao\fire_detect\tree\runs\segment\predict47/'  # 预测图片目录
original_dir = r'D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-04\qie/'  # 原始图片目录
output_dir = r'D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-04\output/'  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 固定HSV阈值参数（蓝色范围）
H_MIN = 100
H_MAX = 140
S_MIN = 50
S_MAX = 255
V_MIN = 50
V_MAX = 255

# 形态学处理核
kernel = np.ones((7, 7), np.uint8)

# 获取预测目录中的所有jpg文件
predict_images = glob.glob(os.path.join(predict_dir, '*.png'))
print(predict_images)
for predict_path in predict_images:
    # 构建对应原始图片路径
    filename = os.path.basename(predict_path)
    print(filename)
    original_path = os.path.join(original_dir, filename)

    if not os.path.exists(original_path):
        print(f"跳过未找到的原始图片: {original_path}")
        continue

    # 读取图片对
    image = cv2.imread(predict_path)
    image2 = cv2.imread(original_path)

    if image is None or image2 is None:
        print(f"图片读取失败: {predict_path} 或 {original_path}")
        continue

    try:
        # 统一尺寸
        image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 生成蓝色区域掩膜
        lower = np.array([H_MIN, S_MIN, V_MIN])
        upper = np.array([H_MAX, S_MAX, V_MAX])
        mask = cv2.inRange(hsv, lower, upper)

        # 检查掩膜是否为空
        if cv2.countNonZero(mask) == 0:
            print(f"跳过无有效掩膜的图片: {filename}")
            continue

        # 形态学处理
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 应用掩膜进行分割
        segmented = cv2.bitwise_and(image2, image2, mask=mask)

        # 保存结果
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, segmented)
        print(f"成功处理: {filename}")

    except Exception as e:
        print(f"处理失败 {filename}: {str(e)}")

print("批量处理完成！")