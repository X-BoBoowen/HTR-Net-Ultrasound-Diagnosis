import cv2

video = cv2.VideoCapture("D:/waibao/yemian_detect/data2/Clipchamp.mp4")  # 打开视频文件
frame_number = 1  # 记录读取到的帧数，初始值为1
frame_interval = 100  # 每隔100帧（截取1帧）
while (video.isOpened()):  # 视频文件被打开后
    retval, frame = video.read()  # 读取视频帧
    if retval == True:  # 读取到视频帧后
        if (frame_number % frame_interval == 0):  # 每隔100帧
            print("jietu")
            cv2.imwrite("D:/waibao/yemian_detect/data11/" + str(frame_number) + ".jpg", frame)  # 截取并保存1帧
    else:  # 没有读取到视频帧
        break  # 终止循环八月份
    frame_number = frame_number + 1  # 读取到的视频帧执行自加操作
    cv2.waitKey(1)  # 1毫秒后播放视频文件的下一帧
print("视频帧已截取完成！")  # 控制台输出提示信息
video.release()  # 释放被视频文件占用的空间