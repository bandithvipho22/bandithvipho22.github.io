---
title: "MHFusionNet: Multiple Hypotheses Fusion-Based Approach For 3D Human Pose Estimation"
excerpt: "<br/><img src='/images/MH_Intro.png'>"
collection: portfolio-1
---

![MHFusionNet](/images/MH_Intro.png)

# Table of Contents
- [ABSTRACT](#Abstract)
- [I. Introduction](#i-introduction)
    - [1.1. Research Background](#11-Research-Background)
    - [1.2. Applications for 3D Human Pose Estimation](#12-Applications)
    - [1.3. Problem of Statement](#13-Problem-statement)
    - [1.4. Objection](#14-Objective)
    - [1.5. Scope of works](#15-scope-of-works)
- [II. Proposed Method](#ii-Proposed-Method)
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

## 1.2. Applications for 3D Human Pose Estimation
3D Human Pose Estimation (HPE) provides geometric and motion information of the human body and can be applied to a wide range of applications such as video animation, human-computer interactions (HCI) [5], action recognition [6], health care of elderly patients[7], gesture recognition [8], and video surveillance [9]. 3D HPE can be applied to a wide range of applications such as:

![mh_3dhp](/images/MH_application.png)

<p align="center"><em>Figure: Real-World Application 3D HPE</em></p>

## 1.3. Problem of Statement
2D-to-3D lifting in human pose estimation remains an ill-posed problem due to depth ambiguity and occlusions in 2D inputs. While transformer-based methods like PoseFormer [10] attempt to reduce information loss, many approaches still predict only a single hypothesis, often failing under occlusion. To address this, recent works such as DiffPose [13], DDHPose [14], and MHFormer [12] generate multiple hypotheses, which improves accuracy but still faces limitations. For example, ManiPose [3] averages multiple hypotheses, resulting in uncertainty and unreliable predictions.

![mh_3dhp_problem](/images/MH_problem.png)

<p align="center"><em>Figure: Some of the SOTA method [12], given a frame with occlued body parts</em></p>

More advanced approaches like D3DP [1] introduce Joint-wise Reprojection-based Multi-hypothesis Aggregation (JPMA) using diffusion models, achieving strong performance. However, JPMA depends on intrinsic camera parameters for reprojection, making it less practical for real-world applications.

## 1.4. Objective
To address the problem that mentioned above, this work proposed MHFusionNet, the Multiple Hypotheses Fusion-Based Approach for 3D Human Pose Estimation. The proposed MHFusionNet designed to integrate multiple 3D human pose hypotheses into a final prediction. Unlike prior approaches that rely on averaging (ManiPose[3]), oracle-based selection, or intrinsic camera parameters [1]. By leveraging a learnable fusion network, MHFusionNet is designed to handle pose ambiguity and occlusion more effectively, while remaining practical
and applicable in real-world scenarios where camera calibration data is unavailable.

## 1.5. Scope of works
The main contributions are summarized as follows:
- Implemen a diffusion-based model to generate multiple 3D pose hypotheses from 2D keypoints using a pre-trained model as backbone.
- Design a fusion network that takes the multi-hypotheses as input to predicts the final 3D human pose.
- Train and evaluate our MHFusionNet by following standard benchmarks (MPJPE).
- Perform a comparative analysis of the proposed MHFusionNet against baseline approaches and SOTA technique to assess accuracy and robustness.

# 2. Proposed Method
An overview of the proposed MHFusionNet is illustrated in Figure below, the architecture consists of two stages: 
- The Multiple Hypotheses Network (Freezed).
- The Proposed Multiple Hypotheses Fusion Network.

![mh_intro_1](/images/MH1.png)

<p align="center"><em>Figure: Overview of the proposed MHFusionNet Method</em></p>

In the first stage, we adopt the *D3DP model* as the baseline to generate multiple plausible 3D human pose hypotheses from a 2D input. These diverse hypotheses are then passed to the second stage, the Fusion Network (FN) which is specifically trained to identify and synthesize the most accurate final 3D human pose from the given set of
the hypotheses. 

The proposed ***MHFusionNet*** leverages a pre-trained multi-hypotheses model in the first stage to generate multiple 3D human pose. In this second stage, the FN was designed based on two strategies Feature Fusion (FF) and Early Fusion (EF) techniques.
This approach advances upon prior state-of-the-art (SOTA) methods by modeling uncertainty more effectively, rather than relying on simple assumptions like averaging.

## 2.1. Multiple Hypotheses Generator
To generate diverse and plausible 3D human pose predictions, we utilize the D3DP model [1], a diffusion-based framework designed for 3D pose estimation. Unlike traditional models that produce a single deterministic output, D3DP can generate multiple hypotheses that reflect the inherent uncertainty and ambiguity in 2D-to-3D pose lifting, especially in cases of occlusion or visually similar joint configurations.

Diffusion-based 3D Human Pose Estimation is the method Diffusion-based with Joint-wise reprojection-based Multi-hypothesis Aggregation (JPMA). On the other hand, D3DP generates multiple possible 3D pose hypotheses for a single 2D observation. It gradually diffuses the ground truth 3D poses to a random distribution and learns a denoiser conditioned on 2D keypoints to recover the uncontaminated 3D poses. The proposed D3DP is compatible with existing 3D pose estimator. JPMA is proposed to assemble multiple hypotheses generated by D3DP into a single 3D pose for practical use. It reprojects 3D pose hypotheses to the 2D camera plane, selects the best hypotheses to the 2D camera plane, selects the best hypothesis joint-by-joint based on the reprojection errors, and combines the selected joints into the final pose.

![mh_intro_1](/images/MH_d3dp.png)

<p align="center"><em>Figure: The overview of D3DP for multiple hypotheses generation</em></p>

The proposed *JPMA* conduct aggregation at the joint level and makes use of the 2D prior information, both of which have been overlooked by previous approaches. D3DP used a mixed spatial temporal Transformer-based method, as the backbone.

As shown in the Figure above, the D3DP model uses a denoising diffusion process to generate multiple diverse 3D human pose hypotheses from a Gaussian distribution. The main idea is to start with random noise and progressively denoise it using a trained model conditioned on 2D keypoints. As we can see in Figure 2.2, D3DP begins by sampling the noise vectors from a standard Gaussian distribution. These noisy vectors are treated as corrupted 3D poses and are fed into a denoiser 𝐷, which is conditioned on the 2D keypoints 𝑥 and
timestep 𝑡. The denoiser learns to reconstruct the clean 3D pose hypotheses $\hat{y}_0$, this process can be expressed as:

$$
\hat{y}_0 = D(y_t, x, t) \tag{1}
$$

Where other processes will follow by formula (Eq. 2, 3, 4, 5) in D3DP [1], to generate multiple hypotheses, we repeat the sampling process $H$ times from a Gaussian distribution $N(0, I)$, giving us:

$$
\tilde{Y} = \{ \tilde{y}_0^{(1)}, \tilde{y}_0^{(2)}, \dots, \tilde{y}_0^{(H)} \} \tag{2}
$$

Each hypothesis $\hat{y}_0 \in \mathbb{R}^{J\times 3}$  , where $J$ is the number of joints. Optionally, this process can be refined over $K$ iteration ($K$: number of samplings timestep) using the DDIM
strategy [15], which improves the accuracy of the generated hypotheses.

## 2.2. Feature Fusion
Feature Fusion is fusing the individual feature representations from each hypothesis, by taking the 3D input and mapping them into a high-dimensional space to obtain the feature.

![mh_FF](/images/MH_FF.png)

<p align="center"><em>Figure: The overview of Feature Fusion (FF)</em></p>

As shown in Figure above, the Feature Fusion consists of two modules: 
- Feature Extractor (FE)
- Regression Head (RH)

The process begins with the multiple 3D human pose as the input passed through the Feature Extractor that converts the raw data from the 3D pose into high dimensional feature. These features encode spatial-temporal
information of the body joints and the dynamics across frames.
