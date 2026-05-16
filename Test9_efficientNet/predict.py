# import os
# import json
#
# import torch
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
#
# from model import efficientnet_b0 as create_model
#
#
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     img_size = {"B0": 224,
#                 "B1": 240,
#                 "B2": 260,
#                 "B3": 300,
#                 "B4": 380,
#                 "B5": 456,
#                 "B6": 528,
#                 "B7": 600}
#     num_model = "B0"
#
#     data_transform = transforms.Compose(
#         [transforms.Resize(img_size[num_model]),
#          transforms.CenterCrop(img_size[num_model]),
#          transforms.ToTensor(),
#          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
#     # load image
#     img_path = r"D:\WeChat\WeChat Files\wxid_irdzyttbtypt22\FileStorage\File\2025-04\avi\data\train/label0/1-4.jpg"
#     assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
#     img = Image.open(img_path)
#     plt.imshow(img)
#     # [N, C, H, W]
#     img = data_transform(img)
#     # expand batch dimension
#     img = torch.unsqueeze(img, dim=0)
#
#     # read class_indict
#     json_path = './class_indices.json'
#     assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
#
#     with open(json_path, "r") as f:
#         class_indict = json.load(f)
#
#     # create model
#     model = create_model(num_classes=2).to(device)
#     # load model weights
#     model_weight_path = "./weights/model-99.pth"
#     model.load_state_dict(torch.load(model_weight_path, map_location=device))
#     model.eval()
#     with torch.no_grad():
#         # predict class
#         output = torch.squeeze(model(img.to(device))).cpu()
#         predict = torch.softmax(output, dim=0)
#         predict_cla = torch.argmax(predict).numpy()
#
#     print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
#                                                  predict[predict_cla].numpy())
#     plt.title(print_res)
#     for i in range(len(predict)):
#         print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
#                                                   predict[i].numpy()))
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()


import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import efficientnet_b0 as create_model


def process_folder(input_folder, output_folder, model, device, class_indict, img_size):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 定义预处理
    resize_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size)
    ])

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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

            # 生成结果文本
            result_text = f"{class_indict[str(predict_cla)]} ({predict[predict_cla].numpy():.2f})"

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 配置参数
    img_size = 224  # B0模型
    input_folder = r"D:\waibao\two-stream-pytorch-master\two-stream-pytorch-master\test"  # 修改为输入文件夹路径
    output_folder = r"D:\waibao\two-stream-pytorch-master\two-stream-pytorch-master\test\model1_label0"  # 修改为输出文件夹路径

    # 加载类别信息
    with open('./class_indices.json', "r") as f:
        class_indict = json.load(f)

    # 初始化模型
    model = create_model(num_classes=10).to(device)
    model.load_state_dict(torch.load("./weights/model-99.pth", map_location=device))
    model.eval()

    # 处理整个文件夹
    process_folder(input_folder, output_folder, model, device, class_indict, img_size)


if __name__ == '__main__':
    main()