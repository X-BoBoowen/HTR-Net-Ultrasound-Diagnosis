import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import efficientnet_b0 as create_model
import pandas as pd
from datetime import datetime
import re


def process_folder(input_folder, output_folder, model, device, class_indict, img_size, correction_df=None):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 创建DataFrame来存储结果
    results_df = pd.DataFrame(columns=["Image Name", "Predicted Class", "Confidence", "Timestamp", "Corrected"])

    # 定义预处理
    resize_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size)
    ])

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 如果有校正数据，准备一个映射字典
    correction_dict = {}
    if correction_df is not None:
        print("Building correction dictionary...")
        for _, row in correction_df.iterrows():
            # 处理各种可能的index格式
            index_val = str(row['index']).strip()
            # 移除可能的'.0'后缀（如果Excel中的数字被存储为浮点数）
            if index_val.endswith('.0'):
                index_val = index_val[:-2]
            correction_dict[index_val] = int(row['label'])
        print(f"Correction dictionary built with {len(correction_dict)} entries")
        print("Sample entries:", list(correction_dict.items())[:5])

    # 遍历输入文件夹
    for filename in os.listdir(input_folder):
        # 只处理图片文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, filename)
        try:
            # 打开并预处理图片
            img_pil = Image.open(img_path).convert('RGB')
            img_display = resize_transform(img_pil)
            img_tensor = tensor_transform(img_display)
            img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

            # 模型预测
            with torch.no_grad():
                output = torch.squeeze(model(img_tensor)).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # 获取预测结果
            predicted_class_idx = predict_cla
            confidence = predict[predict_cla].numpy()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            corrected = False

            # 从文件名中提取index
            # 尝试多种可能的格式
            file_index = None
            # 尝试匹配纯数字前缀
            match = re.match(r'^(\d+)', filename)
            if match:
                file_index = match.group(1)
            else:
                # 尝试匹配其他常见格式，如 "img_123.jpg"
                match = re.search(r'(\d+)', filename)
                if match:
                    file_index = match.group(1)

            # 检查是否需要校正
            if file_index and file_index in correction_dict:
                true_label = correction_dict[file_index]

                # 强制使用Excel中的label值作为预测结果
                predicted_class_idx = true_label
                corrected = True
                print(f"Corrected prediction for {filename}: original {predict_cla} -> {true_label}")
            elif file_index:
                print(f"No correction found for {filename} with index {file_index}")

            predicted_class = class_indict[str(predicted_class_idx)]

            # 添加到DataFrame
            new_row = pd.DataFrame({
                "Image Name": [filename],
                "Predicted Class": [predicted_class],
                "Confidence": [confidence],
                "Timestamp": [timestamp],
                "Corrected": [corrected]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            # 生成结果文本
            result_text = f"{predicted_class} ({confidence:.2f})"
            if corrected:
                result_text += " [Corrected]"

            # 保存结果图片
            plt.figure(figsize=(8, 8))
            plt.imshow(img_display)
            plt.title(result_text, fontsize=12)
            plt.axis('off')

            output_path = os.path.join(output_folder, f"pred_{filename}")
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close()

            print(f"Processed: {filename} -> {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    # 保存Excel文件
    excel_path = os.path.join(output_folder, "prediction_results.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"Prediction results saved to: {excel_path}")

    # 计算平均置信度
    calculate_average_confidence(excel_path, os.path.join(output_folder, "average_confidence_results.xlsx"))

    return results_df


def calculate_average_confidence(input_excel, output_excel):
    # 读取Excel文件
    df = pd.read_excel(input_excel)

    # 提取前缀（如103-1.jpg的前缀是103）
    df['Prefix'] = df['Image Name'].apply(lambda x: re.match(r'^(\d+)', x).group(1) if re.match(r'^(\d+)', x) else None)

    # 按前缀分组并计算平均置信度
    grouped = df.groupby('Prefix')['Confidence'].mean().reset_index()
    grouped.columns = ['Prefix', 'Average Confidence']

    # 计算每个前缀的图片数量
    count_df = df.groupby('Prefix').size().reset_index(name='Image Count')

    # 合并结果
    result_df = pd.merge(grouped, count_df, on='Prefix')

    # 保存结果到新的Excel文件
    result_df.to_excel(output_excel, index=False)
    print(f"Average confidence results saved to: {output_excel}")

    return result_df


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 配置参数
    img_size = 224  # B0模型
    input_folder = r"D:\waibao\cancer_detect\data_model1\all-data"  # 修改为输入文件夹路径
    output_folder = r"D:\waibao\cancer_detect\data_model1\train\newlabel0"  # 修改为输出文件夹路径
    correction_file = r"D:\waibao\cancer_detect\gndexdata.xlsx"  # 包含index和label的Excel文件路径

    # 加载类别信息
    with open('./class_indices.json', "r") as f:
        class_indict = json.load(f)
    print("Class indices:", class_indict)

    # 加载校正数据
    correction_df = None
    if os.path.exists(correction_file):
        correction_df = pd.read_excel(correction_file)
        print("Correction data columns:", correction_df.columns.tolist())
        print("First few rows of correction data:")
        print(correction_df.head())

        # 确保列名正确
        if 'index' not in correction_df.columns or 'label' not in correction_df.columns:
            print("Warning: Correction file does not have 'index' or 'label' columns. Proceeding without correction.")
            correction_df = None
        else:
            print("Loaded correction data")
    else:
        print("Correction file not found, proceeding without correction")

    # 初始化模型
    model = create_model(num_classes=len(class_indict)).to(device)
    model.load_state_dict(torch.load("./weights/model-2.pth", map_location=device))
    model.eval()

    # 处理整个文件夹
    process_folder(input_folder, output_folder, model, device, class_indict, img_size, correction_df)


if __name__ == '__main__':
    main()