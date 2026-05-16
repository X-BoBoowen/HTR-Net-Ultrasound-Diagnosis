import torch

from model import efficientnet_b0 as create_model

if __name__ == '__main__':
    device = torch.device("cpu")
    model = create_model(num_classes=4).to(device)
    model.load_state_dict(torch.load("weights/model-99.pth", map_location=torch.device('cpu')))
    inputs = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, (inputs), "./classify-test.onnx", verbose=True)
    print("模型转换成功!")