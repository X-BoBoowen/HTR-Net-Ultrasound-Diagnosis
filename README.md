# HTR-Net: Multi-Dimensional Fusion Framework for Ultrasound Tumor Diagnosis and Segmentation

An end-to-end ultrasound image tumor diagnosis system addressing three core challenges: blurred lesion boundaries, high miss rates for small tumors, and underutilization of clinical biomarkers. HTR-Net integrates improved YOLOv11 semantic segmentation, EfficientNet-based classification, and a GDX ensemble learning model for multi-dimensional fusion of imaging and clinical data.

## Method Overview

- **Stage 1** — Improved YOLOv11 with C3k2_DFF (Dual-Domain Feature Fusion) and IEMA attention for precise lesion localization → **85.7% mIoU**
- **Stage 2** — TR-Net based on EfficientNet-B0 with transfer learning for binary malignancy classification → **~85% accuracy**
- **Stage 3** — GDX ensemble model (XGBoost + GradientBoosting stacking + Random Forest meta-classifier) fusing clinical features → **AUC 0.88**

## Repository Structure

```
├── Test9_efficientNet/               # Image classification + XGBoost + fusion + utilities
│   ├── model.py                      #   EfficientNet B0–B7 architecture
│   ├── train.py                      #   EfficientNet training script
│   ├── utils.py                      #   Data loading, training/eval utilities
│   ├── predict.py                    #   Batch image prediction
│   ├── new-predict.py               #   Prediction with label correction
│   ├── pth-onnx.py                  #   PyTorch → ONNX export
│   ├── XGBOOST.py                    #   Bayesian-optimized XGBoost training + visualization
│   ├── XGBOOST-predict.py           #   XGBoost inference with preprocessing pipeline
│   ├── data_confuse.py              #   Multi-modal fusion (confidence-weighted + RF)
│   ├── confuse.py                   #   Confusion matrix visualization
│   ├── data_relavate.py             #   Feature correlation & violin plot analysis
│   ├── excle_process.py             #   Excel data matching
│   └── image_require.py            #   Video frame extraction
│
├── ultralytics-main-cancer/          # YOLOv11 segmentation + post-processing
│   ├── train.py                      #   YOLO segmentation training
│   ├── predict.py                    #   YOLO inference
│   ├── area_calculate.py            #   Lesion area measurement
│   ├── length_calculate.py          #   Skeleton-based lesion length
│   ├── width_calculate.py           #   PCA-based lesion width
│   ├── data_exter.py                #   HSV-based mask extraction
│   ├── lianghua.py                  #   Interactive HSV threshold analysis
│   ├── json-txt.py                  #   LabelMe JSON → YOLO TXT conversion
│   ├── filerename.py                #   Batch file renaming
│   ├── file_remove.py               #   XML label batch update
│   ├── extract-data.py              #   YOLO training metrics extraction
│   └── npz.py                       #   NPZ feature extraction → Excel
│
├── fig_classification_segmentation/  # Classification & segmentation result figures
├── fig_clinical_experiments/         # Clinical experiment figures
├── fig_paper/                        # Paper figures (architecture, overview)
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Results

### Ablation Study (8 Groups)

| Group | DFF | IEMA | SCDown | mIoU | Params (M) | GFLOPs |
|-------|-----|------|--------|------|------------|--------|
| N1 | | | | 77.40 | 2.01 | 5.60 |
| N2 | ✓ | | | 80.35 | 2.32 | 5.95 |
| N3 | | ✓ | | 79.82 | 2.28 | 5.88 |
| N4 | | | ✓ | 79.56 | 2.05 | 5.52 |
| N5 | ✓ | ✓ | | 82.17 | 2.59 | 6.23 |
| N8 | ✓ | ✓ | ✓ | **85.25** | **2.08** | **5.80** |

Three-component synergy delivers **+7.85pp mIoU** with fewer parameters than two-component variants.

### Multi-Strategy Comparison

| Strategy | Train AUC | Val AUC | Sensitivity | Specificity |
|----------|-----------|---------|-------------|-------------|
| GLS (clinical only) | 0.7039 | 0.6705 | 0.7125 | 0.7208 |
| TR-Net (image only) | 0.8503 | 0.8181 | 0.7660 | 0.7720 |
| **HTR-Net (fusion)** | **0.8800** | **0.8478** | **0.8680** | 0.7480 |

### Benchmark Comparison

HTR-Net outperforms four published multimodal models (MultimodalCKDModel, CHAIDDecisionTree, BreastNoduleModel, HaTU-Net) on all 7 evaluation metrics.

## Setup

```bash
pip install -r requirements.txt
```

## Dependencies

- PyTorch / torchvision — deep learning framework
- Ultralytics — YOLO object detection & segmentation
- XGBoost — gradient boosting classifier
- scikit-learn — ML utilities
- OpenCV — image processing
- SHAP — model interpretability
- scikit-optimize — Bayesian hyperparameter tuning

## Dataset & Model Weights

- **Ultrasound images**: ~14,000+ multi-view ultrasound images from CAMUS public dataset and 4 medical centers (350 clinical cases). Raw images are excluded from this repository due to patient privacy and size constraints.
- **Trained model weights** (`.pth`, `.pkl`, `.onnx`) are available upon request or via cloud storage link.
- Sample data and all training/evaluation code are fully provided for reproducibility.

## Citation

If you find this work useful, please cite:

```bibtex
@article{xue2025htrnet,
  title={HTR-Net: A Multi-Dimensional Fusion Framework for Ultrasound Image Tumor Diagnosis},
  author={Xue, Bowen},
  journal={Under review at IEEE BIBM 2026},
  year={2025}
}
```

## License

For academic research purposes only. Clinical data involves patient privacy and may not be redistributed without authorization.
