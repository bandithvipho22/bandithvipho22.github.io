---
title: "Sim-To-Real Reinforcement Learning for Robot Navigation"
excerpt: "<br/><img src='/images/sim_to_real.png'>"
collection: portfolio
---

![Sim-To-Real Reinforcement Learning](/images/sim_to_real.png)

---

# I. Introduction

## 1.1. What is Reinforcement Learning?

Reinforcement learning is for the agent to learn how to make better decisions by choosing actions that maximize rewards over time. This RL loop outputs a sequence of state, action, reward and next state.

![Reinforcement Learning Loop](/images/rl_01.png)

## 1.2. Study Background
This project inspire from "Tommaso Van Der Meer" which is using reinforcement learning for robot navigation in "Hospital World" gazebo, ROS2 frameworks for simulation.

![Reinforcement Env](/images/rl_env.png)

In Environment of "Hospitalbot", including some features with difference scenarios for training RL, such as:
- Random Robot location
- Random Target: 
- Fix Robot location
- Fix Target

### 1.3.1.Random Robot Location

## 1.3. Problem of Statement

**Challenges in Robot Navigation**:

- Real-world navigation is complex due to noise, obstacles, and dynamics.
- Training RL model directly in real world can cause: Large amount of real-world data, Time consuming to collect, Hardware damage, Safety risks, hard to manage and control.

**Why sim-to-real Reinforcement Learning**:
- Sim-to-real enables training in simulation and transferring the knowledge to real
robot.

![Wifibot Env](/images/wifibot.png)

In figure above, illustrated the Wifibot research for navigation using sim-to-real reinforcement learning. Here is the research paper-based ([Resource](https://arxiv.org/pdf/2004.14684))

## 1.4. Objective

- Self-navigate from current position to a predefined target point with unknown Environment by using Reinforcement Learning.
- Sim-To-Real Reinforcement Learning with Differential Robot Robot movement visualization Performance of RL model in real robot visualization.

# 