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
This project inspire from "[Tommaso Van Der Meer](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning)" which is using reinforcement learning for robot navigation in "Hospital World" gazebo, ROS2 frameworks for simulation.

![Reinforcement Env](/images/rl_env.png)

In Environment of "[Hospitalbot](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning/tree/humble/hospital_robot_spawner/worlds)", including some features with difference scenarios for training RL that exist on "[hospital_robot_spawner/hospitalbot_env.py](https://github.com/TommasoVandermeer/Hospitalbot-Path-Planning/blob/humble/hospital_robot_spawner/hospital_robot_spawner/hospitalbot_env.py)", including some features:
- Random Robot location
- Random Target
- Fix Robot location
- Fix Target

**Random Robot Location:**

Ramdom Robot Location means, the robot’s location is randomized at the beginning of each episode. This means that when an episode ends (e.g., the robot reaches the target, hits an obstacle, or exceeds a time limit), the environment resets, and a new episode begins with the robot placed at a new, randomly selected starting position.
- Hit an Obstacle: defined as a terminal state in the environment (e.g., a failure condition), this would end the current episode. The environment resets, and the robot is placed at a new random location for the next episode.
- Reach Target: Similarly, reaching the target typically ends the episode (as a success condition). When the next episode starts, the robot’s starting position is again chosen randomly.

Ramdom Robot Location, encourages the RL agent to learn the policy that generalizes across different starting points. It prevents the agent from overfitting to a specific starting point and promotes robustness in navigation.

**Random Target:**

The target location (e.g., a patient room, a delivery point, or a charging station) is randomly selected at the start of each episode from a set of possible target locations. 
- The target location is randomized at the beginning of each episode, similar to the random robot location. When an episode ends (e.g., the robot reaches the target or fails by hitting an obstacle), the environment resets, and a new target location is chosen randomly for the next episode.
- Hit obstacle or Reach target: These events typically terminate the episode, and a new random target is set for the next episode.

**Fix Robot and Fix Target Location:**

The robot always starts each episode from the same, predetermined location in the hospital environment (e.g., a fixed charging station or a central hub). Example: The robot always starts at the hospital’s main entrance or a designated charging station, regardless of the episode.

---
In this work, as illustrated in the figure below, the RL agent is trained within a Gazebo simulation environment. The robot’s initial position is fixed (**blue dot**), while the target positions are randomized (**red dots**). These randomized target points are generated along a path, ensuring that the agent learns to navigate under diverse conditions rather than memorizing a single trajectory. This setup improves generalization, enabling the robot to adapt to different goal locations while maintaining robustness in path planning.

![RL Way-Points](/images/rl_waypoint.png)

---
**Software Requirement:**
- Ubuntu 22.04
- ROS2 Humble
- Gazebo & RViz
- gymnasium library
- stable baseline framework
- pytorch
- Micro-ROS: used for communication between PC and Micro-controller (Arduino MEGA). Link for setup ([LINK](https://github.com/micro-ROS/micro_ros_arduino/tree/humble))

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

![rl objective](/images/rl_obj.png)

## 1.5. Scope of Works
 