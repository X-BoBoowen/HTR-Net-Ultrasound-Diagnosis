import os


def rename_images(folder_path):
    """
    重命名文件夹中所有 *msrcp.jpg 格式的图片文件
    将文件名中的 '_msrcp' 部分移除，保留原始文件名主体
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否符合模式：以 "_msrcp.jpg" 结尾
        if filename.endswith("_msrcp.jpg"):
            # 构建新文件名：移除 "_msrcp" 部分
            new_filename = filename.replace("_msrcp", "")

            # 构建完整的文件路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # 执行重命名
            os.rename(old_path, new_path)
            print(f"重命名成功: {filename} -> {new_filename}")


# 使用示例
folder_path = r"D:\waibao\low_bright_image_enhance\val\msrcp"  # 替换为你的实际文件夹路径
rename_images(folder_path)