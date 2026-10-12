# Real-Time Attention-Guided Object Detection and Dynamic Feature Fusion for Dense Unstructured Traffic in Developing Regions

**Md. Taher Bin Omar Hijbullah¹, Md. Rony Mia¹, Mahedi Hasan¹\*, Md. Rashedul Islam¹, Utpal Kanti Das¹**

¹ *Department of Computer Science and Engineering, IUBAT—International University of Business Agriculture and Technology, Dhaka 1230, Bangladesh*
\* *Corresponding Author: mahedi.hasan@iubat.edu*

## Abstract

Automated visual perception in urban transportation systems is critical for Intelligent Transportation Systems (ITS) and Advanced Driver Assistance Systems (ADAS). However, state-of-the-art vision models typically demonstrate a steep drop in reliability when migrated from structured Western highways to the chaotic, unstructured traffic corridors of South Asian megacities. Bangladeshi road environments are typified by heterogeneous multimodal traffic compositions—co-mingling high-speed motorized vehicles with non-motorized transport such as human-powered rickshaws, vans, and three-wheeled auto-rickshaws (CNGs)—exhibiting complete lane indiscipline, extreme visual occlusion, and severe class imbalance. To overcome these perception bottlenecks while maintaining strict edge-grade real-time viability, this study presents a comprehensively enhanced YOLOv11 framework tailored for complex urban traffic perception. We introduce two synergistic architectural enhancements: (i) an optimized Convolutional Block Attention Module (CBAM) embedded strategically at the deep feature extraction stage (Layer 23, P5 feature map) to sequentially reweight channel and spatial saliency, effectively suppressing background visual clutter while sharpening the structural boundaries of occluded vehicles, and (ii) a Dynamic Channel Fusion Module (DCFM) equipped with parallel multi-scale convolutional receptive fields and adaptive feature weighting to resolve scale discrepancies across diminutive and bulky road participants. To rigorously benchmark the framework, we curated and standardized a multi-source regional benchmark containing 265,698 high-resolution annotated bounding box instances across nine localized classes, implementing video-wise sequence partitioning to strictly prohibit temporal data leakage. Extensive ablation experiments over 50 epochs demonstrate that our attention-guided YOLOv11m architecture attains 75.70% mAP@0.50 (53.2% mAP@0.50:0.95), achieving an outstanding 90.8% mAP@0.50 on the visually intricate Rickshaw class and boosting minority Truck accuracy by +1.5%, all while executing at an ultra-low inference latency of 4.7 ms per frame (~212.7 FPS on NVIDIA T4), which actually outperforms baseline execution speed. Crucially, a rigorous error analysis isolating cross-class structural confusion reveals that architectural enhancements alone cannot compensate for severe long-tail scarcity in minority categories (e.g., Cycles), establishing foundational empirical guidelines for future ITS dataset curation. Finally, the framework was validated through a containerized real-time edge deployment prototype, confirming its operational suitability for scalable smart-city traffic monitoring.

**Keywords:** Real-time object detection, YOLOv11, Convolutional Block Attention Module (CBAM), Dynamic Channel Fusion Module (DCFM), Unstructured traffic, Intelligent Transportation Systems (ITS), Edge-grade latency, Class imbalance.

---

## 1. Introduction

*(See formatted sections in DOCX manuscript for complete text and equations)*

## 2. Related Work

## 3. Proposed Methodology and Mathematical Formulation

## 4. Dataset Engineering and Experimental Setup

## 5. Experimental Results and Comparative Analysis

## 6. In-Depth Error Isolation, Confusion Matrix, and Structural Limitations

## 7. Real-Time Edge Deployment and Practical Prototype Validation

## 8. Conclusion and Future Research Directions

## References

- [1] Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? The KITTI vision benchmark suite. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3354–3361 (2012)
- [2] Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The Cityscapes dataset for semantic urban scene understanding. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213–3223 (2016)
- [3] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: European Conference on Computer Vision (ECCV), pp. 740–755. Springer (2014)
- [4] Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779–788 (2016)
- [5] Woo, S., Park, J., Lee, J.Y., Kweon, I.S.: CBAM: Convolutional block attention module. In: European Conference on Computer Vision (ECCV), pp. 3–19. Springer (2018)
- [6] Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7132–7141 (2018)
- [7] Jocher, G., Chaurasia, A., Qiu, J.: YOLOv11: Real-time object detection and segmentation. Ultralytics (2024). https://github.com/ultralytics/ultralytics
- [8] Khanam, R., Hussain, M.: YOLOv11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725 (2024)
- [9] Islam, M.M., Rashid, M.R.A., Das, A., Hasan, M.R., Shams, K., Hasan, K.F.: Bangladeshi traffic flow dataset. Mendeley Data, V6 (2025). https://doi.org/10.17632/h8bfgtdp2r.6
- [10] Shihavuddin, A., Rashid, M.R.A.: DhakaAI: Real-world autonomous driving dataset for Bangladesh. Harvard Dataverse (2020). https://doi.org/10.7910/DVN/POREXF
- [11] Sujoy, M.I.M., Rahman, M.A., Dhruba, U.S., Prattoy, S.E.: Monocular depth-aware object detection for irregular traffic: The Poribohon-BD dataset. BRAC University Technical Report (2024)
- [12] Hasan, M.M., Wang, Z., Hussain, M.A.I., Fatima, K.: Bangladeshi native vehicle classification based on transfer learning with deep convolutional neural network. Sensors 21(22), 7545 (2021)
- [13] Rahman, M.S., Ullah, H., Khan, F.: Fine-tuning deep convolutional networks for local vehicle classification. In: International Conference on Innovations in Science, Engineering and Technology (ICISET), pp. 421–426. IEEE (2022)
- [14] Rahman, R., Bin Azad, Z., Hasan, M.B.: Densely-populated traffic detection using YOLOv5 and non-maximum suppression ensembling. In: International Conference on Big Data, IoT, and Machine Learning (BIM), pp. 567–578. Springer (2021)
- [15] Garshasebi, N.: Improving small object detection in remote sensing with YOLO11-CBAM and deep learning. Diqiu Kexue 49(2), 112–125 (2025)
- [16] Khan, A., Ali, S., Rahim, A.: Defect detection in industrial pipelines using attention-guided YOLOv11. IEEE Access 13, 45210–45224 (2025)
- [17] Zhang, X., Yu, X., Feng, J., Chen, W., Li, Y.: TriNext: An efficient three-path fusion module for multi-scale feature enhancement in object detection. Science Progress 108(4), 00368504251395116 (2025)
- [18] Hossain, G.F., Kabir, S., Zabbar, J., Islam, M.M., Jesmin, S.: Visual assessment of the rickshaw pullers in Dhaka city. Journal of Ophthalmology and Research 7(3), 57–62 (2024)
- [19] Hijbullah, M.T.B.O., Mia, M.R., Hasan, M., Islam, M.R., Das, U.K.: Attention-guided YOLOv11 for object detection in unstructured Bangladeshi traffic. In: IEEE International Conference on Engineering and Frontiers of Technology (ICEFronT), IEEE (2026)
