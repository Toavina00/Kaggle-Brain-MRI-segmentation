# Brain MRI Tumor Segmentation using Deep Learning

## 📌 Overview
Implementation of brain tumor segmentation using four pre-trained CNN architectures (ResNet50, EfficientNet-B2, DenseNet121, ShuffleNet V2) on the [LGG MRI Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation). The models predict binary masks to identify tumor regions in MRI scans.

## 🛠️ Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- pandas
- scikit-learn
- kagglehub (v0.3.10+)

## 📦 Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) (233 MRI scans + masks)
- **Split**: 65% train / 15% validation / 20% test
- **Preprocessing**:
  - Resize to model-specific dimensions:
    - ResNet/DenseNet/ShuffleNet: 224x224
    - EfficientNet-B2: 260x260
  - Normalize with mean `[0.5, 0.5, 0.5]` and std `[0.5, 0.5, 0.5]`

## 📊 Models
| Model          | Test Loss | Architecture Note                          |
|----------------|-----------|--------------------------------------------|
| ResNet50       | 0.0117    | FC layer: 2048 → 224x224 + Sigmoid         |
| EfficientNet-B2| 0.0106    | Classifier: 1408 → 260x260 + Sigmoid        |
| DenseNet121    | 0.0105    | Classifier: 1024 → 224x224 + Sigmoid        |
| ShuffleNet V2  | 0.0193    | FC layer: 1024 → 224x224 + Sigmoid         |

## 🚀 Training Details
- **Epochs**: 30
- **Optimizer**: Adam (LR=0.001, β=(0.9, 0.999))
- **Scheduler**: Exponential LR decay (γ=0.9)
- **Loss Function**: Binary Cross-Entropy

## 📈 Results
- **Best Performance**: DenseNet121 (lowest test loss)
- **Lightweight Option**: EfficientNet-B2 (balance of performance/efficiency)
- **Room for Improvement**: ShuffleNet V2
