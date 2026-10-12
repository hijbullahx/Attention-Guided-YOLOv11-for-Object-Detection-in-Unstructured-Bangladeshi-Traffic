# Springer Nature Journal LaTeX Package (JRTIP Submission)

**Paper Title:** Real-Time Attention-Guided Object Detection and Dynamic Feature Fusion for Dense Unstructured Traffic in Developing Regions  
**Target Journal:** *Journal of Real-Time Image Processing* (JRTIP), Springer Nature  
**Authors:** Md. Taher Bin Omar Hijbullah, Md. Rony Mia, Mahedi Hasan*, Md. Rashedul Islam, Utpal Kanti Das  
**Affiliation:** Department of Computer Science and Engineering, IUBAT—International University of Business Agriculture and Technology, Dhaka, Bangladesh  

---

## 🚀 How to Open and Compile on Overleaf

1. **Option A (Instant 1-Click Upload):**
   - Download the pre-built zip archive located at:  
     `Springer_Journal_Manuscript/Springer_JRTIP_Overleaf.zip`
   - Log in to [Overleaf](https://www.overleaf.com/).
   - Click **New Project** $\rightarrow$ **Upload Project**.
   - Select `Springer_JRTIP_Overleaf.zip`.
   - Overleaf will automatically unzip all files and compile using **pdfLaTeX**.

2. **Option B (Manual Folder Upload):**
   - Upload all contents of this `LaTeX_Overleaf` directory into a new blank project on Overleaf.
   - Set the main document to either `main.tex` or `sn-article.tex`.
   - Set the compiler engine to **pdfLaTeX** (default in Overleaf).
   - Click **Recompile**.

---

## 📁 Project Directory Structure

```text
LaTeX_Overleaf/
│
├── main.tex                    # Primary Springer Nature LaTeX manuscript
├── sn-article.tex              # Duplicate entry-point alias for Overleaf template recognition
├── sn-bibliography.bib         # BibTeX references with complete citation metadata
├── sn-jnl.cls                  # Official Springer Nature LaTeX class file (v2.1)
│
├── sn-mathphys.bst             # Springer Nature Math/Physical Sciences reference style
├── sn-basic.bst                # Springer Nature Basic reference style
├── sn-vancouver.bst            # Springer Nature Vancouver reference style
├── sn-aps.bst                  # Springer Nature APS reference style
│
├── figures/                    # High-resolution publication figures
│   ├── fig1_western_vs_dhaka.png       (Domain gap: Western highway vs Dhaka congestion)
│   ├── fig2_cbam_architecture.png      (CBAM Channel + Spatial Attention modules)
│   ├── fig3_proposed_pipeline.png      (YOLOv11m + CBAM Layer 23 complete architecture)
│   ├── fig4_dcfm_architecture.png      (Dynamic Channel Fusion Module)
│   ├── fig5_dataset_distribution.png   (Long-tailed distribution across 9 traffic classes)
│   ├── fig6_pr_curves.png              (Precision-Recall curves: baseline vs. proposed)
│   ├── fig7_confusion_matrix.png       (Normalized confusion matrix & error isolation)
│   ├── fig8_qualitative_results.png    (Ground truth vs. YOLOv11m+CBAM qualitative detections)
│   └── fig9_streamlit_prototype.png    (Real-time edge surveillance Streamlit prototype)
│
└── README.md                   # This documentation guide
```

---

## ⚙️ Layout Configurations (`main.tex`)

The manuscript is pre-configured with:
```latex
\documentclass[pdflatex,sn-mathphys]{sn-jnl}
```
If you wish to switch between layout variants, adjust the documentclass options:
* **Single Column (Default Springer Submission):** `\documentclass[pdflatex,sn-mathphys]{sn-jnl}`
* **Double Column (Typeset Article Layout):** `\documentclass[default,iicol,sn-mathphys]{sn-jnl}`
* **Double Line Spacing (Referee Review Mode):** `\documentclass[referee,sn-mathphys]{sn-jnl}`
* **With Line Numbers:** `\documentclass[lineno,sn-mathphys]{sn-jnl}`

---

## 📊 Summary of Main Experimental Results

* **Overall Accuracy:** Attains **75.70% mAP@0.50** (+0.60% net boost over YOLOv11m baseline) and **53.20% mAP@0.50:0.95**.
* **Indigenous Rickshaw Accuracy:** Reaches **90.8% mAP@0.50**.
* **Minority Truck Boost:** Boosts underrepresented Truck detection by **+1.50%** (from 45.5% to 47.0%).
* **Edge Inference Latency:** Executes at **4.7 ms per frame (~212.7 FPS)** on an NVIDIA Tesla T4 GPU, which is faster than baseline execution (5.0 ms / 200.0 FPS) due to attention-guided background suppression reducing NMS bounding box overhead.
