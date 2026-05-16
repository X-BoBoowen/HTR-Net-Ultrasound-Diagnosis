# 多维度融合的超声图像肿瘤智能诊断系统

基于超声图像和临床数据的多模态融合肿瘤智能诊断系统，整合深度学习图像分类、YOLO语义分割和XGBoost临床特征分析的智能辅助诊断方案。

## 项目结构

```
├── requirements.txt          # Python 依赖
├── README.md                 # 项目说明
├── .gitignore                # Git 忽略规则
│
├── efficientnet/             # 模态一：超声图像分类 (EfficientNet-B0)
│   ├── model.py              #   EfficientNet B0-B7 模型定义
│   ├── train.py              #   训练脚本
│   ├── utils.py              #   数据加载、训练/评估工具
│   ├── predict.py            #   批量图片预测
│   └── pth-onnx.py           #   PyTorch 转 ONNX 导出
│
├── xgboost/                  # 模态二：临床特征分类 (XGBoost)
│   ├── XGBOOST.py            #   贝叶斯优化的 XGBoost 训练 + 可视化
│   └── XGBOOST-predict.py    #   模型预测 + 预处理管线保存
│
├── yolo_seg/                 # 模态三：超声图像语义分割 (YOLOv11-Seg)
│   ├── train.py              #   YOLO 分割模型训练
│   ├── predict.py            #   YOLO 分割推理
│   ├── data_exter.py         #   从分割掩膜提取目标区域
│   └── lianghua.py           #   交互式 HSV 阈值量化分析
│
├── fusion/                   # 多模态融合
│   └── data_confuse.py       #   置信度加权融合 + 随机森林融合
│
├── analysis/                 # 特征分析与可视化
│   ├── data_relavate.py      #   特征相关性分析 (点二列/XGBoost/琴谱图)
│   └── confuse.py            #   混淆矩阵可视化
│
├── segmentation_post/        # 分割后处理与量化测量
│   ├── area_calculate.py     #   病变区域面积计算
│   ├── length_calculate.py   #   骨架法裂缝/病变长度测量
│   └── width_calculate.py    #   PCA 法裂缝/病变宽度测量
│
├── utils/                    # 数据处理工具
│   ├── excle_process.py      #   Excel 数据匹配
│   ├── extract-data.py       #   从 YOLO 训练结果提取指标
│   ├── file_remove.py        #   XML 标注文件名批量更新
│   ├── filerename.py         #   批量文件重命名
│   ├── json-txt.py           #   LabelMe JSON → YOLO TXT 格式转换
│   ├── npz.py                #   NPZ 特征提取 → Excel
│   └── image_require.py      #   视频帧截取
│
├── data/                     # 示例数据 (不含原始数据集)
│   ├── data.xlsx             #   临床特征数据
│   ├── results.csv           #   训练结果
│   ├── 分割数据.csv           #   分割评估数据
│   └── ...
│
└── figures/                  # 论文结果图
    ├── 不同模态的分类模型精度对比图.png
    ├── 分割效果图展示.png
    ├── 混淆矩阵-XGBOOST.png
    ├── 相关性热力图.png
    └── ...
```

## 研究方法

### 三模态融合诊断框架

| 模态 | 方法 | 输入 | 用途 |
|------|------|------|------|
| 模态一 | EfficientNet-B0 | 超声图像 | 图像级良恶性分类 |
| 模态二 | XGBoost + 贝叶斯优化 | 临床特征 (LV-GLS, LVEF, etc.) | 临床指标分类 |
| 模态三 | YOLOv11-Seg | 超声图像 | 病灶区域语义分割 |

多模态预测结果通过置信度加权融合和随机森林融合得到最终诊断。

## 环境配置

```bash
# 创建虚拟环境
conda create -n cancer_detect python=3.10
conda activate cancer_detect

# 安装依赖
pip install -r requirements.txt
```

## 数据集

本项目使用甲状腺/乳腺超声图像数据集和临床特征数据。

### 数据获取
- 超声图像数据集：约 14,000+ 张，包含多视角超声图像
- 临床特征数据：包含年龄、LV-GLS、LVEF、LAVImax 等指标
- 原始数据集因涉及医院隐私数据，不直接包含在代码仓库中
- 示例数据文件位于 `data/` 目录

### 数据准备
```
data/
├── train/
│   ├── label0/      # 良性样本
│   └── label1/      # 恶性样本
└── val/
    ├── label0/
    └── label1/
```

## 使用说明

### 1. 图像分类 (EfficientNet)
```bash
cd efficientnet
python train.py --data-path ./data/train --epochs 100 --batch-size 16
python predict.py   # 批量预测
```

### 2. 临床特征分类 (XGBoost)
```bash
cd xgboost
python XGBOOST.py           # 训练 + 可视化
python XGBOOST-predict.py   # 预测新数据
```

### 3. 语义分割 (YOLOv11)
```bash
cd yolo_seg
python train.py     # 训练分割模型
python predict.py   # 分割推理
```

### 4. 多模态融合
```bash
cd fusion
python data_confuse.py
```

## 模型权重

训练好的模型权重文件因体积较大未包含在仓库中：

| 模型 | 文件 | 说明 |
|------|------|------|
| EfficientNet | model-99.pth | 图像分类模型 |
| XGBoost | final_model.pkl | 临床特征分类模型 |
| YOLO | best.pt | 语义分割模型 |
| ONNX | classify-test.onnx | 分类模型 ONNX 导出 |

## 依赖库

- **PyTorch / torchvision**: 深度学习框架
- **Ultralytics**: YOLO 目标检测与分割
- **XGBoost**: 梯度提升树分类器
- **scikit-learn**: 机器学习工具
- **OpenCV**: 图像处理
- **SHAP**: 特征重要性可解释性分析
- **scikit-optimize**: 贝叶斯超参数优化

## License

学术研究用途。数据集涉及患者隐私，未经授权不得分发。
