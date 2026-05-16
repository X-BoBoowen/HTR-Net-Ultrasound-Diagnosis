# import os
# import shutil
#
#
# def collect_files(source_folder, target_folder):
#     """
#     收集源文件夹中所有子文件夹的图片和XML文件到目标文件夹
#     新文件名格式: 原文件名_子文件夹名.扩展名
#     """
#     # 支持的图片扩展名列表
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
#     # 支持的XML扩展名
#     xml_extension = '.xml'
#
#     # 创建目标文件夹（如果不存在）
#     os.makedirs(target_folder, exist_ok=True)
#
#     # 遍历源文件夹中的所有子文件夹
#     for root, dirs, files in os.walk(source_folder):
#         # 获取当前子文件夹名称
#         current_folder = os.path.basename(root)
#
#         # 跳过源文件夹本身（只处理子文件夹）
#         if root == source_folder:
#             continue
#
#         # 处理当前子文件夹中的所有文件
#         for file in files:
#             file_path = os.path.join(root, file)
#             filename, extension = os.path.splitext(file)
#             extension = extension.lower()  # 统一转为小写
#
#             # 检查是否为图片或XML文件
#             if extension in image_extensions or extension == xml_extension:
#                 # 构建新文件名：原文件名_子文件夹名.扩展名
#                 new_filename = f"{filename}_{current_folder}{extension}"
#                 target_path = os.path.join(target_folder, new_filename)
#
#                 # 复制文件到目标文件夹
#                 shutil.copy2(file_path, target_path)
#                 print(f"已复制: {file} -> {new_filename}")
#
#
# if __name__ == "__main__":
#     # 配置路径
#     source_folder = r"D:\xwechat_files\wxid_irdzyttbtypt22_bd48\msg\file\2025-07\labelling\labelling"  # 替换为你的源文件夹路径
#     target_folder = r"D:\xwechat_files\wxid_irdzyttbtypt22_bd48\msg\file\2025-07\labelling\output"  # 替换为你的目标文件夹路径
#
#     # 执行收集操作
#     collect_files(source_folder, target_folder)
#     print("\n所有文件已处理完成！")


import os
import xml.etree.ElementTree as ET


def update_xml_filenames(folder_path):
    """
    修改指定文件夹中所有XML文件的<filename>标签内容
    新内容为：XML文件名（不含扩展名） + 原图片扩展名
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.xml'):
            xml_path = os.path.join(folder_path, filename)

            try:
                # 解析XML文件
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # 查找<filename>标签
                filename_tag = root.find('filename')
                if filename_tag is not None:
                    # 获取原文件名的扩展名（保留原图片格式）
                    original_name = filename_tag.text
                    if '.' in original_name:
                        # 保留最后一个点后面的扩展名
                        extension = '.' + original_name.split('.')[-1]
                    else:
                        # 如果没有扩展名，默认使用.jpg
                        extension = '.jpg'

                    # 获取XML文件名（不含扩展名）
                    xml_basename = os.path.splitext(filename)[0]

                    # 构建新的文件名：XML基本名 + 原扩展名
                    new_filename = f"{xml_basename}{extension}"

                    # 更新<filename>标签内容
                    filename_tag.text = new_filename

                    # 保存修改后的XML
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    print(f"已更新: {filename} -> <filename>{new_filename}</filename>")
                else:
                    print(f"警告: {filename} 中未找到<filename>标签，已跳过")

            except ET.ParseError:
                print(f"错误: {filename} 不是有效的XML文件，已跳过")
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    # 设置包含XML文件的文件夹路径
    folder_path = r"D:\xwechat_files\wxid_irdzyttbtypt22_bd48\msg\file\2025-07\labelling\output"  # 替换为你的XML文件夹路径

    # 执行更新操作
    update_xml_filenames(folder_path)
    print("\n所有XML文件处理完成！")