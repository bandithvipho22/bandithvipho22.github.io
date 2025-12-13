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

***Notice***: You explore about my research through here [[View My Report](MHFusionNet.pdf)].

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

The process begins with the multiple 3D human pose as the input passed through the Feature Extractor that converts the raw data from the 3D pose into high dimensional feature. These features encode spatial-temporal information of the body joints and the dynamics across frames.

### 2.2.1. Feature Extractor
Feature Fusion extracts high-dimensional features representations from each hypothesis independently, then integrates them through fusion mechanism. By motivating from the FusionFormer [3], in the Feature Extractor We adopt PoseFormer [13] as the feature extractor in the main experiments.

Our Feature extractor is built upon a Poseformer-based architecture, which was originally designed to operate on 2D input keypoints in spatial-temporal. In our ***MHFusionNet***, we modified the **Poseformer** [13] to accept 3D input, allowing it to extract high-dimensional feature embeddings from each individual 3D pose hypothesis. The overall procedure of our feature extractor is illustrated in ***Figure Below***.

![mh_FF](/images/MH_FE.png)

<p align="center"><em>Figure: Overview of Feature Extractor (FE)</em></p>

After obtaining the set of multiple 3D human pose from the pre-trained model D3DP model, we have an input tensor $H_i$ of shape [B, F, J, 3], where 3 denotes the ($x, y, z$) coordinates:

$$
F_{3D} : H_i \in \mathbb{R}^{B \times F \times J\times 3} \tag{3}
$$

Each input $H_i$ contains $H$ Hypotheses across $F$ frame, yielding an overall set that denotes the estimation results as $P_{3D}$:

$$
P_{3D} = F_{3D}(I) \in \mathbb{R}^{B \times F \times H \times J \times 3} \tag{4}
$$

The D3DP model takes an input $I$ (2D observations), $F_{3D}(I)$ denoted the D3DP model's inference yielding $H$ hypotheses of 3D pose across $F$ frames as shown in (Eq. 4). At this point, $P_{3D}$ contains a set of hypotheses 3D poses for each frame across the batch represented by B.

To extract high-dimensional feature embeddings, $P_{3D}$ is passed through the Feature Extractor, which projects each 3D point into latent space of high dimensionality. This can be represented:

$$
F_{embed} = Embed(P_{3D}) \mathbb{R}^{B \times F \times H \times J \times C_H} \tag{5}
$$

Where $C_H$ is the number of channels of each keypoint. Subsequently, the feature extractor employs several layers to extract the relationship between keypoint. Each joint of every hypothesis and frame is embedded into this latent space, allowing the model to learn spatial and temporal relationships across the pose sequence.

To embedded features $F_{embed}$ are further processed by the Pose Feature Encoder $E_{pose}$ , which applied a series of attention mechanisms as in Poseformer-based, extracting the relationship between joints, frame, and hypotheses:

$$
F^0_{pose} = E_{pose}(F_{embed}) \in \mathbb{R}^{B \times F \times H \times J \times C_H} \tag{6}
$$

Where:
- $P_{3D}$: Output of the D3DP model across $F$ frames and $H$ hypotheses.
- $F_{embed}$: High-dimensional embeddings of the 3D input $P_{3D}$.
- $F_{pose}$: Final feature representation learned by the pose feature encoder.

$F_{pose}$, capture both the spatial structure within each hypothesis and the temporal consistency across frames. They are then passed to the regression head to produce the final refined 3D pose prediction.

### 2.2.2. Regression Head
To maximize the capacity of our fusion network, we employ a simple 3D pose regression head to map the fused hypotheses features into final 3D human pose predictions. Once we have the encoded feature map $F_{Pose}$ from the pose feature encoder:

$$
F_{pose} \in \mathbb{R}^{B \times F \times H \times J \times C_H} \tag{7}
$$

The goal of the regression head is to aggregate information across the $H$ hypotheses and regress to a single final 3D human pose prediction. Before we pass the $F_{pose}$ to the regression head, first we concatenate the feature embeddings across the hypothesis dimension $H$ . This creates a combined feature for each joint across all hypotheses:

$$
F_{concat} = concat(F_{pose}) \in \mathbb{R}^{B \times F \times J \times (H \times C_{H})} \tag{8}
$$ 

As we can see in (Eq. 8) above, we concatenated the 𝐹𝑝𝑜𝑠𝑒 across hypotheses on $C_H$ channels. This concatenation allows the model to access information from all hypotheses simultaneously, making it possible to learn inter-hypotheses relationships and select the best parts of each hypothesis.

After obtaining $F_{concat}$ , we pass it to the Regression Head (RH) to produce the final 3D human pose prediction. The regression head (RH) acts as a mapping network that takes the combined feature space and learns to regress it to valid 3D pose:

$$
\tilde{P}_{3D} = R_{\theta} (F_{concat}) \in \mathbb{R}^{B \times F \times J \times 3} \tag{9}  
$$

Where $R_{\theta} (.)$ is the regression head implemented with learnable parameters $\theta$ and $\tilde{P}_{3D}$ is the Final 3D human pose prediction, matching the ground truth shape ($B, F, J, 3$). 

To investigate the best way to map from the high-dimensional $F_{concatenate}$ space to the final 3D pose, we implemented and experimented with different network structures for our regression head, including MLP, ResidualFC, and DenseFC. Each approach operates on the input feature of shape (𝐻 × 𝐶). The details of each design below.

### a. Multi-Layer Perceptron (MLP)
MLP is used as a neural network to model the relationship between inputfatures and a continuous output variable. MLP is a feadforward network comprised of a sequence of Linear and activation functions. In this network, we constructed three layers by using Linear module and ReLU activation functions as shown in Figure below.

![mh_mlp](/images/MH_mlp.png)

<p align="center"><em>Figure: MLP Network</em></p>


