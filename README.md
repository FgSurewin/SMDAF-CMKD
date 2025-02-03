# SMDAF: A Scalable Sidewalk Material Data Acquisition Framework with Bidirectional Cross-Modal Knowledge Distillation


> This repository contains the official code for the paper "SMDAF: A Scalable Sidewalk Material Data Acquisition Framework with Bidirectional Cross-Modal Knowledge Distillation" accepted at the 2025 Winter Conference on Applications of Computer Vision (WACV 2025).


![SMDAF](./images/2025%20WACV%20Framework%20Diagram_small_3.png)

## Abstract

Ensuring safe and independent navigation poses considerable difficulties for individuals who are blind or have low vision (BLV), as it requires detailed knowledge of their immediate environment. Our research highlights the critical need for accessible data on sidewalk materials and objects, which is currently lacking in existing map services. To bridge this gap, we present the Sidewalk Material Data Acquisition Framework (SMDAF), designed for large-scale data collection. This framework includes (1) a lightweight data collection system embedded in a white cane, which captures audio data through the interaction of the cane tip with the sidewalk surface, and a mobile app that facilitates data storage and management, resulting in a novel multimodal dataset comprising both image and audio data; and (2) a unique Cross-Modal Knowledge Distillation (CMKD) technique for an enhanced audio material classifier. Our CMKD approach employs an image-based model as the teacher to improve the audio model, incorporating an Enhanced Bidirectional learning method with an intuitive filtering technique: Bidirectional Correct Sample Filtering (BCSF). BCSF filters correct samples to prevent the distillation of incorrect knowledge, addressing the issue of inaccurate cross-modal learning. This novel approach has resulted in a 1.84\% improvement in Macro Accuracy, achieving an overall accuracy of 87.62\%, surpassing all state-of-the-art KD and CMKD methods. This study underscores the efficacy of SMDAF and provides a practical CMKD technique for future cross-modal learning tasks.


## Open-Source Multimodal Sidewalk Material Dataset
Download the multimodal sidewalk material dataset from the following link: [SMDAF Dataset](https://drive.google.com/file/d/1B8-mlAN58YIJ-R847B7q6FeWFLzgP84p/view?usp=sharing)