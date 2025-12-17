# 🗂️ ROADMAP — Polyp Detection in Colonoscopy Images

**Project:** Polyp Detection in Colonoscopy Images Using Deep Learning
**Team:** Percepta (Azimjon Akromov, Sanjar Raximjonov)
**Course:** Computer Vision (Fall 2025)
**Institution:** Central Asian University

---

## 📅 Week-by-Week Milestones

| Week | Dates | Milestone / Task | Deliverable | Owner(s) |
|------|--------|------------------|--------------|-----------|
| **W1** | Oct 14–21 | Team formation, topic confirmation, and repo setup | GitHub repo with README.md and ROADMAP.md created | All |
| **W2** | Oct 21–27 | Literature review and dataset preparation | Summary of related papers; Kvasir-SEG & CVC-ClinicDB datasets downloaded and processed | Azimjon + Sanjar |
| **W3** | Oct 28–Nov 3 | Baseline models implementation (YOLOv8 & Mask R-CNN) | Training scripts; initial mAP results on validation set | Azimjon |
| **W4** | Nov 4–10 | Model training and optimization | Trained YOLOv8 and Mask R-CNN models with evaluation metrics | Azimjon + Sanjar |
| **W5** | Nov 11–17 | Evaluation & Explainability | Full evaluation vs. baselines; Grad-CAM visualizations | Azimjon |
| **W6** | Nov 18–24 | GUI development & results analysis | Interactive web interface (Gradio); comprehensive results table | Sanjar |
| **W7** | Nov 25–Dec 1 | Final report writing & presentation prep | Final project report (PDF); presentation slides | Azimjon + Sanjar |
| **W8** | Dec 2–8 | Final refinements & presentation | Code cleanup, final model weights, defense preparation | All |

---

## ✅ Weekly Progress Log

Use this section to log updates as you work each week.
Each update should include **3–6 short bullet points** about progress, challenges, or changes.

### Week 1 (Oct 14–21)
- [x] Formed Team Percepta (Azimjon Akromov, Sanjar Raximjonov)
- [x] Selected Project: Polyp Detection in Colonoscopy Images
- [x] Created and populated GitHub repository with initial documentation

### Week 2 (Oct 21–27)
- [x] Completed literature review on object detection and polyp detection in medical imaging
- [x] Downloaded and processed Kvasir-SEG and CVC-ClinicDB datasets
- [x] Prepared dataset splits (1,289 training, 323 validation images)

### Week 3 (Oct 28–Nov 3)
- [x] Set up YOLOv8 and Mask R-CNN training scripts
- [x] Implemented data augmentation pipeline (flips, color jittering, mosaic)
- [x] Began training baseline models

### Week 4 (Nov 4–10)
- [x] Completed YOLOv8 training (20 epochs, mAP@0.5: 0.16)
- [x] Completed Mask R-CNN training (10 epochs, mAP@0.5: 0.001)
- [x] Analyzed training curves and convergence behavior

### Week 5 (Nov 11–17)
- [x] Ran comprehensive evaluation on validation set for all models
- [x] Generated comparative result tables (mAP@0.5, mAP@0.5:0.95)
- [x] Created Grad-CAM explainability visualizations
- [x] Analyzed success and failure cases

### Week 6 (Nov 18–24)
- [x] Developed interactive web GUI using Gradio
- [x] Implemented real-time detection interface with model selection
- [x] Analyzed why YOLOv8 outperformed Mask R-CNN
- [x] Wrote core sections of final report (Method, Experiments, Results)

### Week 7 (Nov 25–Dec 1)
- [x] Finalized comprehensive PDF report with all sections
- [x] Generated automated report using `generate_report_pdf.py`
- [x] Prepared presentation slides outline
- [x] Created Q&A defense preparation materials

### Week 8 (Dec 2–8)
- [x] Code cleanup and documentation
- [x] Final model weights saved and organized
- [x] Defense preparation (Q&A guide, cheat sheet)
- [ ] Final presentation and defense

---

## 👥 RACI Matrix

| Task | Responsible | Accountable | Consulted | Informed |
|------|--------------|--------------|------------|-----------|
| Repo setup & documentation | Azimjon | All | — | Instructor |
| Literature review & Data Prep | Azimjon + Sanjar | Azimjon | — | Instructor |
| Baseline Model Implementation | Azimjon | Sanjar | — | Team |
| Model Training & Optimization | Azimjon + Sanjar | Azimjon | — | Team |
| Evaluation & Visualization | Azimjon | All | — | Instructor |
| GUI Development | Sanjar | Azimjon | — | Team |
| Report & Presentation | Azimjon + Sanjar | All | — | Instructor |

---

## 🚀 Deliverables Summary

- ✅ **Final Report:** Comprehensive PDF report (`experiments/results/final_report.pdf`) documenting methodology, results, and analysis
- ✅ **Trained Models:** YOLOv8 model weights achieving mAP@0.5 of 0.16, Mask R-CNN baseline
- ✅ **Interactive GUI:** Web-based interface (`src/new_gui.py`) for real-time polyp detection
- ✅ **Evaluation Results:** Complete metrics, visualizations, and training curves
- ✅ **Explainability:** Grad-CAM visualizations showing model attention
- ✅ **Code Repository:** Fully functional codebase with training, evaluation, and visualization scripts
- ✅ **Defense Materials:** Q&A preparation guide, presentation outline, and cheat sheet
- ✅ **Documentation:** Updated README.md and ROADMAP.md with project details

## 📊 Key Achievements

- **YOLOv8 Performance:** mAP@0.5 of 0.1597 (160x better than Mask R-CNN)
- **Real-time Inference:** <1 second per image, suitable for clinical deployment
- **Interactive Interface:** Gradio-based GUI with model selection and parameter tuning
- **Comprehensive Analysis:** Success/failure case analysis, explainability visualizations
- **Complete Documentation:** Professional PDF report with all sections

---

## 📝 Notes

- All major milestones completed successfully
- YOLOv8 demonstrated superior performance compared to Mask R-CNN
- Interactive GUI enables clinical testing and demonstration
- Comprehensive evaluation framework established for future work

---

> *Maintained by Team Percepta – Central Asian University, Fall 2025*