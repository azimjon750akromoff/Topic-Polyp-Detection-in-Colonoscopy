# FewShot-Polyp-Detection
**Team Percepta’s project on few-shot object detection for medical imaging — enabling polyp detection with minimal labeled data.**

---

## 🧠 Project Overview
This project explores **Few-Shot Object Detection** for identifying polyps in colonoscopy images. In medical settings, labeled data is scarce and expensive to acquire. We aim to develop a system that can learn to detect polyps from only **5 examples per class** (5-shot), using **prompt-tuned Vision Transformers (ViTs)** and meta-learning techniques.

Our research focuses on bridging the gap between data-efficient learning and critical medical applications, providing a robust baseline for low-data scenarios.

---

## 👥 Team Members
| Name | Student ID | Email | Role |
|------|-------------|--------|------|
| Azimjon Akromov | 220291 | 220291@centralasian.uz | Model architecture & repo management |
| Sanjar Raximjonov | 220304 | 220304@centralasian.uz | Experiments & evaluation metrics |

---

## 🎯 Objectives
- Implement a **few-shot object detection** model for polyp detection using prompt-tuned ViTs.
- Compare against strong baselines like **YOLO** and **Mask R-CNN** under the same low-data regime.
- Evaluate the model's performance using standard object detection metrics (mAP, F1-score).
- Provide visual explanations of the model's predictions to aid interpretability.

---

## 📚 Dataset

We will use publicly available colonoscopy polyp detection datasets:

- **[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)** — Contains 1,000 polyp images with corresponding bounding boxes and segmentation masks.
- **[CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)** — 612 images from 31 colonoscopy sequences with polyp annotations.

These datasets are standard benchmarks for polyp detection and segmentation tasks.

---

## ⚙️ Methodology
1.  **Baseline:** Train standard object detection models (**YOLOv8**, **Mask R-CNN**) in a 5-shot setting on the Kvasir-SEG dataset.
2.  **Few-Shot Approach:** Implement a **prompt-tuned ViT** detector (inspired by OWLViT) or a **meta-learning** approach (e.g., FSRW) adapted for the 5-shot scenario.
3.  **Evaluation:** Compare the few-shot model against the baselines using mean Average Precision (mAP), F1-score, and inference speed.
4.  **Explainability:** Generate attention maps or Grad-CAM visualizations to understand the model's focus areas.

---

## 🧪 Experiments & Evaluation
| Experiment | Description | Metric |
|-------------|--------------|--------|
| Baseline YOLO | Standard YOLOv8 fine-tuned on 5-shot data | mAP@0.5, F1 |
| Baseline Mask R-CNN | Standard Mask R-CNN fine-tuned on 5-shot data | mAP@0.5, F1 |
| Proposed Few-Shot Model | Prompt-tuned ViT / Meta-learning for 5-shot detection | mAP@0.5, F1 |
| Comparison | Performance and efficiency vs. baselines | mAP, F1, Speed |

---

## 🧭 Roadmap
*See `ROADMAP.md` for a detailed and updated week-by-week plan.*

| Week | Milestone | Owner |
|------|------------|--------|
| Week 1 | Team formation & topic selection | All |
| Week 2 | Related work summary + dataset setup | Azimjon + Sanjar |
| Week 3 | Baseline YOLO & Mask R-CNN (5-shot) | Azimjon |
| Week 4 | Few-shot model implementation | Sanjar |
| Week 5 | Evaluation & explainability | Azimjon |
| Week 6 | Final report draft & results analysis | All |
| Week 7 | Final proposal submission & repo polishing | All |

🗂️ **ROADMAP.md** file will include weekly progress updates and issue tracking.

---

## 🛠️ Tech Stack
- Python 3.10
- PyTorch, PyTorch Lightning
- Ultralytics (for YOLO), Detectron2 (for Mask R-CNN)
- HuggingFace Transformers / OpenCLIP
- Matplotlib / Seaborn for visualization
- Google Colab Pro / Kaggle GPU runtime

---

## ⚖️ Ethics & Compliance
- The Kvasir-SEG and CVC-ClinicDB datasets are **publicly available** for research use.
- This project involves medical data; we will adhere to strict usage guidelines and cite the original data sources.
- The goal is to assist clinicians, not replace them. Results will be presented with appropriate caveats.

---

## 📈 Expected Outcomes
- A working prototype of a few-shot polyp detection model.
- Comparative analysis report vs. standard baselines in low-data settings.
- Final deliverables: Code, trained model weights, and a comprehensive project report.

---

## 🧩 References
- [1] J. et al. OWLViT: Simple Open-Vocabulary Object Detection with Vision Transformers. *ECCV 2022*.
- [2] Kang, B., et al. Few-Shot Object Detection via Feature Reweighting. *ICCV 2019*.
- [3] Pogorelov, K., et al. Kvasir-SEG: A Segmented Polyp Dataset. *MMSys 2017*.
- [4] Redmon, J., & Farhadi, A. YOLOv3: An Incremental Improvement. *arXiv 2018*.
- [5] He, K., et al. Mask R-CNN. *ICCV 2017*.

---

## 📜 License
This project is conducted as part of the **Central Asian University — Computer Vision (Fall 2025)** course under academic fair use for research and educational purposes.

---

## 🌐 Repository Link
[https://github.com/your-username/FewShot-Polyp-Detection](https://github.com/your-username/FewShot-Polyp-Detection) (Replace with your actual repo link)




## First we run train.py ; then eval.py ; then explainability.py - this will get us 2-output results !



### MINE : **W3** | Oct 28–Nov 3 | Baseline models (YOLO & Mask R-CNN) in 5-shot setting | Scripts for 5-shot training; initial mAP results on validation set | Azimjon |



# cd "/Users/azimjonakromov/Desktop/Sanjar + me - Computer vision/Topic-5-Few-Shot-Polyp-Detection-in-Colonoscopy"
# ./venv/bin/python -m src.train
# ./venv/bin/python -m src.eval 
# ./venv/bin/python -m src.explainability