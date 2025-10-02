---
title: "MHFusionNet: Multiple Hypotheses Fusion-Based Approach For 3D Human Pose Estimation"
excerpt: "<br/><img src='/images/MHFusionNet00.png'>"
collection: portfolio-1
---

![MHFusionNet](/images/MHFusionNet00.png)

# Table of Contents
- [ABSTRACT](#Abstract)
- [I. Introduction](#i-introduction)
- [1.1. Objective](#11-objective)
- [1.2. Scope of Work](#12-scope-of-work)
- [II. Methodology](#ii-methodology)
- [III. Result and Discussion](#iii-result-and-discussion)
- [IV. Achievements](#iv-achievements)
- [V. Our Goals](#v-our-goals)

---

# ABSTRACT
In this project, presents the proposed Multiple Hypotheses Fusion-Based
Approach known as a fusion-based for multiple possible 3D human pose hypotheses
estimation. This project contributes a novel framework that addresses ambiguity and
occluding problems in 3D human pose estimation. Most of the State-of-the-Art (SOTA)
developed to the missing depth ambiguity and occlude but there are still limitations
such as the ManiPose [3] produce the multiple hypotheses and used averaging to
compute for final 3D human pose which makes the final 3D pose uncertainty and
unreliable. Recently, D3DP [1] proposed the Joint-wise reprojection-based Multi-
hypothesis Aggregation (JPMA) for probabilistic 3D human pose estimation by using
diffusion-based which achieves exceptional performance. The proposed JPMA
conducts joint-level aggregation based on reprojection errors by relying on intrinsic
camera parameters for projecting 3D pose to the camera plane as a 2D coordinate (u,
v), which is incompatible with real-world applications. Our proposed MHFusionNet,
designed camera-parameter-free approach which is composed of two stages. For the
first stage, leverages a pre-trained multiple hypotheses model to generate multiple 3D
human pose. Second stage, the fusion network was designed based on two strategies
feature fusion (FF) and early fusion (EF) techniques. This approach advances upon
prior state-of-the-art (SOTA) methods by modeling uncertainty more effectively, rather
than relying on simple assumptions like averaging.

# I. Introduction
In recent years, artificial intelligence (AI) has become a driving force behind many
innovations that shape in everyday lives. Among these AI innovations, 3D Human pose
estimation (HPE) is recognized as a key technology which enables machines to perceive and
interpret human movement in real-world applications.

## 1.1. Research Background

3D Human Pose Estimation (3D HPE) predicts human joint positions in 3D space from RGB images or videos, typically in two stages: detecting 2D joint locations and then lifting them to 3D. While effective, this 2D-to-3D lifting is ill-posed due to depth ambiguity, leading to multiple possible solutions. Many existing methods estimate only a single hypothesis, which fails under occlusion, while multi-hypothesis methods often average predictions, reducing reliability.

![mh_3dhp](/images/MH001.png)

<p align="center"><em>Figure: Mainstream approach of 3D Human Pose Estimation</em></p>

Recent work, such as D3DP with JPMA, improves accuracy by aggregating multiple hypotheses using camera parameters, but this limits practicality in real-world settings. To address this, we propose a camera-parameter-free fusion framework that combines multiple hypotheses from diffusion-based models, improving the accuracy and robustness of 3D skeleton prediction in practical applications.

Human Pose Estimation (HPE) provides geometric and motion information of the human body and can be applied to a wide range of applications such as video animation, human-computer interactions (HCI) [5], action recognition [6], health care of elderly patients[7], gesture recognition [8], and video surveillance [9]. 3D HPE can be applied to a wide range of applications such as:

![mh_3dhp](/images/MH_application.png)

<p align="center"><em>Figure: Real-World Application 3D HPE</em></p>

## 1.2. Objective
To address the problem that mentioned above, this work proposed MHFusionNet, the Multiple Hypotheses Fusion-Based Approach for 3D Human Pose Estimation. The proposed MHFusionNet designed to integrate multiple 3D human pose hypotheses into a final prediction. Unlike prior approaches that rely on averaging (ManiPose[3]), oracle-based selection, or intrinsic camera parameters [1]. By leveraging a learnable fusion network, MHFusionNet is designed to handle pose ambiguity and occlusion more effectively, while remaining practical
and applicable in real-world scenarios where camera calibration data is unavailable.

