---
title: "Motion Planning for Mecanum Wheel Robot"
excerpt: "<br/><img src='/images/Motion_Planing.png'>"
collection: portfolio-4
---

![Motion Planning](/images/Motion_Planing.png)

---


# Table of Contents
- [ABSTRACT](#Abstract)
- [I. Introduction](#i-introduction)
- [II. Optimal Control](#ii-Optimal-Control)
- [III. Simulation](#iii-simulation)

---

***Notice***: For full technical details, architecture design, and experimental results, please refer to the complete report: [[View My Report](/files/Motion_planing.pdf)].

---

# ABSTRACT

In this report focuses on Motion Planning for Mobile robots to participat in ABU Robocon Contest. The goal is to study on the motion of these robots move precisely and efficiently using a technique called Non-linear Model Predictive Control (NMPC) with point stabilization. Mecanum wheel robots are known for their ability to move in any direction, but controlling them can be tricky due to their complex movements. We tackle this challenge by developing a smart control system that predicts the robot’s future movements and adjusts its path accordingly. To test our method, we conduct simulations where the robot navigates through various scenarios. Our results show that our approach significantly improves the robot’s ability to move accurately and quickly, which is crucial for success in Robocon. This research offers a practical solution to enhance Mecanum wheel robot performance in competitive settings, making them more competitive and reliable for real-world challenges.

# I. Introduction
## 1. Motion Planning
Motion planning in robotics is the strategy of guiding and determining how a robot
should move to achieve a specific goal, considering obstacles, constraints, and the overall optimization of its motion.

![mh_3dhp](/images/portfolio5_f01.png)

<p align="center"><em>Figure: Overview of Motion Planning</em></p>

In Figure above, Motion planning is like a GPS for robots. Just as a GPS helps you find the best route to your destination, In Figure (b) the motion planning helps robots figure out the best path to move from one point to another without bumping into obstacles. It’s about making smart decisions to navigate safely and efficiently through a complex environment.

## 2. Model Predictive Control (MPC)
Model predictive control is a method that uses a mathematical model of the system
to predict the future behavior of the output and optimize the control input. Unlike traditional control methods that operate based on current states, MPC considers future states and system dynamics over a specified time horizon. By continuously updating predictions and optimizing control actions, MPC enables proactive decision-making to achieve desired system performance
while satisfying constraints.

In Figure below has showm the Basic concept of Model Predictive Control (MPC) that
involved predicting the future behavior of a system based on its current state and using this prediction to optimize control actions.

![motion_planning](/images/portfolio5_f02.png)

<p align="center"><em>Figure: Basic Concept for Model Predictive Control Algorithm</em></p>

There are two Methods of Mathematics Model which can be use to represent the Model Predictive Control (MPC), such as Non-Linear MPC and Linear MPC:
- **Linear Model Predictive Control**: Linear MPC simplifies things by assuming the system follows straight-line rules. This makes it easier to crunch numbers and control sys-
tems with basic behaviors.
- **Non-linear Model Predictive Control**: Nonlinear MPC handles systems with complex
behaviors, like those described by quadratic equations, without needing them to follow
simple rules. It can use nonlinear models and constraints, accurately representing intricate
system behaviors. Nonlinear MPC is more flexible, capturing a wider range of behaviors
than linear MPC.

# II. Optimal Control
## 1. System Model
We decided to choose Mecanum Wheel to use in our system model. The Mecanum wheel, also known as an omni-directional wheel which can move in any direction and adapt to diverse configurations of the platform’s frame.

![motion_planning](/images/portfolio5_f03.png)

<p align="center"><em>Figure: Basic driving directions of omnidirectional where the main wheels are rotating forwards or backwards</em></p>

For detail calculation, or derived the equation of inverse kinematic and Forward Internal Kinematics for Four-Wheel Mecanum Mobile Robot, you can view on my report [[View My Report](/files/Motion_planing.pdf)]

## 2. Non-linear Model Predictive Control
In this work, I used Non-linear Model Predictive Control to implement with the Mecanum wheel robot. In Non-linear Model Predictive Control we work by apply the Quadratic Equation to Solve the Quadratic Cost subject from Optimal Problem to Non-linear functions over the future finite
horizon. To be able applying the Model Predictive Control with Mecanum Wheel Robot, we started with the following step:
- Define Non-linear Equation with Quadratic From.
- Model Predictive Control Formulation.
- Discretirization Method.
- Transform Optimal Control Problem to Non-linear Problem.

# III. Simulation
## 1. Block Control System
This block representation of a Nonlinear Model Predictive Control (NMPC) system, various components work together to achieve control objectives. In the other hand, we can implement the System Model with NMPC in real application by following to this block control system.

![motion_planning](/images/portfolio5_f04.png)

<p align="center"><em>Figure: Block Control System</em></p>

## 2. Simulation setup and Results
In this simulation part, we are simulating by applying the Nonlinear Model Predictive Control with Mecanum Wheel which adjusted:
- N-Predict horizon = 30 at time-step = 0.1
- For end goal (final state): x=10, y=0.0, z=pi/2

![motion_planning](/images/portfolio5_f05.png)

<p align="center"><em>Figure: Conducted Experiment using python and plot result</em></p>

## 3. Conclusion
In conclusion, Non-linear Model Predictive Control (NMPC) is a powerful method for
motion planning in mecanum wheeled robots. Its ability to optimize control actions in real-time,
handle constraints, and adapt to changes makes it a valuable tool. After implement NMPC for mecanum wheel motion planning by using a kinematic model in python code, the simulation shows the robot precisely following a predefined point, showcasing the practicality and effectiveness of NMPC in robotics.

---

***Notice***: For full technical details, architecture design, and experimental results, please refer to the complete report: [[View My Report](/files/Motion_planing.pdf)].

---