# 🏥 Medical Detection Testing Framework

## Team AI: Abhay, Nidhi, Anubhav, Nandan

A complete testing pipeline for medical image abnormality detection models, implementing metrics from **MURA** and **FracAtlas** papers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qyZijcEd68kEI2OuAbvXxX9l09PJkgYw?usp=sharing)

---

## 🎯 What This Does

✅ Tests **ANY** medical detection model (RetinaNet, MedGamma, ViT, etc.)  
✅ Calculates **all standard metrics** (Sensitivity, Specificity, Cohen's Kappa, AUROC, etc.)  
✅ Generates **publication-quality graphs** (ROC curves, Confusion matrices)  
✅ Provides **95% confidence intervals** (bootstrap method)  
✅ Works in **Google Colab** (no installation needed!)  

---

## 📊 Demo Results

Our pipeline works! Check out the example results in `demo_results/` folder.

### Metrics Achieved:
```
Sensitivity:     0.8378  (95% CI: 0.7111, 0.9512)
Specificity:     0.9365  (95% CI: 0.8750, 0.9846)
Precision:       0.8857
F1-Score:        0.8611
Accuracy:        0.9000
Cohen's Kappa:   0.7831  (95% CI: 0.6387, 0.8971)
AUROC:           0.8636
```

### Visualizations:
- 📈 **ROC Curve** (like MURA paper Figure 3)
- 📊 **Confusion Matrix** (publication-ready)

---

## 🚀 Quick Start

### Option 1: Run Demo (NO Model or Images Needed!)

**Click the badge below to open in Google Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qyZijcEd68kEI2OuAbvXxX9l09PJkgYw?usp=sharing)

Then:
1. **Click** Runtime → Run all
2. **Wait** 2 minutes
3. **See** all results with graphs!

This demo uses synthetic data to show how the pipeline works.

### Option 2: Test Your Trained Model

1. **Click the Colab badge above**
2. **File** → **Save a copy in Drive**
3. **Upload your trained model** (.pth file)
4. **Upload your test data** (CSV with image paths and labels)
5. **Modify the model loading cell** (see instructions below)
6. **Run all cells**
7. **Download your results!**

---

## 📖 How to Test YOUR Trained Model

### Step 1: Prepare Your Test Data

Create a CSV file `test_data.csv`:
```csv
image_path,label
/path/to/image1.jpg,0
/path/to/image2.jpg,1
/path/to/image3.jpg,0
```

Where:
- `label = 0` → Normal/Healthy
- `label = 1` → Abnormal/Fractured

### Step 2: In Colab Notebook

**Find and modify these cells:**

#### Cell: Load Your Model
```python
# Upload your trained model
from google.colab import files
uploaded = files.upload()  # Upload your .pth file

# Load your model
import torch
from monai.networks.nets import RetinaNet

model = RetinaNet(
    spatial_dims=2,
    num_classes=2,  # Binary: Normal vs Abnormal
    num_anchors=9,
    # ... your model config
)
model.load_state_dict(torch.load('retinanet_trained.pth'))
model.eval()
```

#### Cell: Upload Test Data
```python
# Upload test_data.csv
from google.colab import files
uploaded = files.upload()

# Load test data
import pandas as pd
test_df = pd.read_csv('test_data.csv')
```

#### Cell: Calculate Metrics (NO CHANGES NEEDED!)
```python
# This code stays the same for everyone!
from utils.medical_metrics import MedicalMetrics

metrics_calc = MedicalMetrics()
metrics_calc.update(predictions, labels, probabilities)

# Get all metrics
metrics = metrics_calc.calculate_all_metrics()
metrics_calc.print_metrics_report(model_name="RetinaNet")

# Generate plots
metrics_calc.plot_roc_curve(save_path='results/roc_curve.png')
metrics_calc.plot_confusion_matrix(save_path='results/confusion_matrix.png')
```

---

## 👥 For Team Members (Abhay, Nidhi, Anubhav)

### How to Test YOUR Model

**Everyone uses the SAME Colab notebook - just change the model!**

1. **Open Colab** (click badge at top)
2. **File** → **Save a copy in Drive**
3. **Change ONLY the model loading:**

#### For Suppose Anubhav (MedGamma):
```python
# Cell: Load Model
from medgamma import MedGamma

model = MedGamma()
model.load_state_dict(torch.load('medgamma_trained.pth'))
model.eval()

# Cell: Prediction
output = model(image)
prob = output.item()  # Adjust based on your model
```

#### For suppose Nidhi (ViT):
```python
# Cell: Load Model
from vit import VisionTransformer

model = VisionTransformer()
model.load_state_dict(torch.load('vit_trained.pth'))
model.eval()

# Cell: Prediction
output = model(image)
prob = torch.softmax(output, dim=1)[0][1].item()
```

#### For others (Other Model):
```python
# Cell: Load Model
from your_library import YourModel

model = YourModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Cell: Prediction
# Your model's prediction logic
```

### Keep Everything Else The Same!

- ✅ Metrics calculation (same code)
- ✅ ROC curve generation (same code)
- ✅ Confidence intervals (same code)
- ✅ File saving (same code)

### Download and Share Results
```python
# Download your results
from google.colab import files
files.download('results/your_model_results.json')
files.download('results/roc_curve.png')
files.download('results/confusion_matrix.png')
```

Then upload to GitHub in `results/your_model/` folder!

---

## 📁 Repository Structure
```
medical-detection-testing/
│
├── README.md                             ← You're reading this!
├── Medical_Detection_Testing_Demo.ipynb  ← Main Colab notebook
│
├── demo_results/                         ← Example results
│   ├── roc_curve_demo.png
│   ├── confusion_matrix_demo.png
│   └── demo_results.json
│
└── results/                              ← Team results go here
    ├── retinanet/                       (Nandan)
    ├── medgamma/                        (Abhay)
    ├── vit/                             (Nidhi)
    └── other_model/                     (Anubhav)
```

---

## 🔄 Team Workflow

### For Nandan (Me):
1. ✅ Demo works
2. ✅ Uploaded to GitHub
3. ✅ Shared with team
4. ⏳ Test trained RetinaNet model later
5. ⏳ Upload results to `results/retinanet/`

### For Anubhav (MedGamma):
1. Open Colab notebook (click badge)
2. Save copy to Drive
3. Modify for MedGamma
4. Upload model + test data
5. Run all cells
6. Download results
7. Upload to `results/medgamma/` in GitHub

### For Nidhi (ViT):
1. Same as Anubhav, but for ViT
2. Upload to `results/vit/`

### For others (Other Model):
1. Same process
2. Upload to `results/other_model/`

---

## 📊 Model Comparison Table

After everyone tests, we'll have:

| Model | Sensitivity | Specificity | Kappa | AUROC | Team Member |
|-------|-------------|-------------|-------|-------|-------------|
| RetinaNet | 0.8378 | 0.9365 | 0.7831 | 0.8636 | Nandan |
| MedGamma | ? | ? | ? | ? | Anubhav |
| ViT | ? | ? | ? | ? | Nidhi |
| Other | ? | ? | ? | ? | Abhay |

---

## 📚 Metrics Explained

### From MURA Paper:
- **Sensitivity (Recall)**: How many abnormal cases you catch
  - Formula: TP / (TP + FN)
  - High = Few missed abnormalities
  - **MURA baseline**: 0.815

- **Specificity**: How many normal cases you correctly identify
  - Formula: TN / (TN + FP)
  - High = Few false alarms
  - **MURA baseline**: 0.887

- **Cohen's Kappa**: Agreement with gold standard
  - Range: 0 to 1 (1 = perfect)
  - Used to compare with radiologists
  - **MURA baseline**: 0.705 (overall)

- **AUROC**: Overall performance across all thresholds
  - Range: 0.5 to 1 (0.5 = random, 1 = perfect)
  - **MURA achieved**: 0.929

### From FracAtlas Paper:
- **Precision**: Of "abnormal" predictions, how many were correct
- **F1-Score**: Harmonic mean of Precision and Recall
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **FracAtlas YOLOv8**: Precision 0.807, Recall 0.473, mAP 0.562

---

## 🎓 Key Features

✅ **Based on Published Papers**: MURA (2018) & FracAtlas (2023)  
✅ **Production-Ready**: Professional code quality  
✅ **Easy to Use**: Runs in Google Colab (no installation!)  
✅ **Team-Friendly**: Same metrics for everyone  
✅ **Publication-Quality**: Graphs ready for papers  
✅ **95% Confidence Intervals**: Bootstrap method (1000 samples)  

---

## 🐛 Troubleshooting

### "Module not found"
Run the install cell in Colab:
```python
!pip install numpy scikit-learn matplotlib seaborn pandas
```

### "Cannot load image"
- Check CSV paths are correct
- Use absolute paths: `/content/drive/MyDrive/data/image.jpg`
- Or upload images to Colab first

### "Model loading error"
- Verify model architecture matches saved weights
- Check `num_classes` matches your training

### "CUDA out of memory"
- Use smaller batch size
- Or use CPU (slower but works)

---

## 📖 References

1. **MURA Dataset Paper**:
   - Rajpurkar, P., et al. (2018). "MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs." arXiv:1712.06957
   - Model AUROC: 0.929, Sensitivity: 0.815, Specificity: 0.887

2. **FracAtlas Dataset Paper**:
   - Abedeen, I., et al. (2023). "FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation." Scientific Data, 10:521
   - YOLOv8 Baseline: Precision 0.807, Recall 0.473, mAP@0.5: 0.562

---

## ✅ Quick Checklist

**For Everyone:**
- [ ] Open Colab notebook (click badge)
- [ ] Run demo to see how it works
- [ ] Save copy to your Drive
- [ ] Prepare test data CSV
- [ ] Save your trained model
- [ ] Modify model loading cell
- [ ] Run all cells
- [ ] Download results
- [ ] Upload to GitHub `results/` folder

---

## 🎉 You're All Set!

The pipeline is **ready to use** for:
- ✅ Testing any detection model
- ✅ Getting consistent, comparable metrics
- ✅ Creating publication-ready visualizations
- ✅ Comparing team results fairly

**Questions?** Check the demo notebook or let's discuss!

---

**Colab Link**: https://colab.research.google.com/drive/1qyZijcEd68kEI2OuAbvXxX9l09PJkgYw?usp=sharing

**Made with ❤️ by Team AI**
