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

<div align="center">

![Reinforcement Env](/images/rl_env.png)

</div>

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

<div align="center">

![RL Way-Points](/images/rl_waypoint.png)

</div>

---
**Software Requirement:**
- Ubuntu 22.04
- ROS2 Humble
- Gazebo & RViz
- OpenAI Gym library
- stable baseline3 framework
- pytorch
- Micro-ROS: used for communication between PC and Micro-controller (Arduino MEGA). Link for setup ([LINK](https://github.com/micro-ROS/micro_ros_arduino/tree/humble))

The robot employed is a Pioneer 3AT with 4-wheel differential drive and a 180° laser for obstacle detection.

## 1.3. Problem of Statement

**Challenges in Robot Navigation**:

- Real-world navigation is complex due to noise, obstacles, and dynamics.
- Training RL model directly in real world can cause: Large amount of real-world data, Time consuming to collect, Hardware damage, Safety risks, hard to manage and control.

**Why sim-to-real Reinforcement Learning**:
- Sim-to-real enables training in simulation and transferring the knowledge to real
robot.

<div align="center">

![Wifibot Env](/images/wifibot.png)

</div>

In figure above, illustrated the Wifibot research for navigation using sim-to-real reinforcement learning. Here is the research paper-based ([Resource](https://arxiv.org/pdf/2004.14684))

## 1.4. Objective

- Self-navigate from current position to a predefined target point with unknown Environment by using Reinforcement Learning.
- Sim-To-Real Reinforcement Learning with Differential Robot Robot movement visualization Performance of RL model in real robot visualization.

<div align="center">

![rl objective](/images/rl_obj.png)

</div>

## 1.5. Scope of Works
In this project we involved with some step:
- Create way-points for Path Planing on mobile robot agent and train it along that path as shown in Figure (a).
- Setup Gazebo world for RL model training base Real World Dimensions (4x15m) as shown in Figure (b), the agent train a long the randomize path in gazebo world.
- Configure gazebo world make sure the environment in gazebo similar to real world (light, floor friction, obstacle, ...).
- Configure robot’s parameters for RL training according to Algobot (Real Robot) like motor speed, torque, friction, and also lidar parameters (range, bit rate, ...).
- Use SAC for training agent in simulation (Gazebo).
- Hardware Implementation -> Deploy RL model into Algobot.

<div align="center">

![scope of work](/images/rl_scopeofwork.png)

</div>

---
# II. Methodology
In this section, we introduce the methodology and setup for training a reinforcement learning (RL) agent using the Soft Actor-Critic (SAC) algorithm via the Gymnasium API. We briefly outline the SAC algorithm, environment configuration, reward structure, and key hyperparameters to optimize agent performance. The section also covers essential hardware/software requirements and preprocessing steps for seamless integration with Gymnasium, providing a concise guide for reproducible RL training.
## 2.1. Soft-Actor Critic (SAC)
SAC concurrently learns a policy $\pi_{\theta}$ and two Q-functions $Q_{\Phi_{1}}, Q_{\Phi_{2}}$, there are 2 variants of SAC that are currently standard: one that uses a fixed entropy regularization coeffient $\alpha$, and another that enforces an entropy constraint by varying $\alpha$ a over the course of training. For
simplicity, Spinning Up makes use of the version with a fixed entropy regularization coefficient,
but the entropy-constrained variant is generally preferred by practitioners.

**Exploration vs Exploitation:**

SAC trains a stochastic policy with entropy regularization, and explores in an on-policy way. The entropy regularization coefficient $\alpha$ explicitly controls the explore-exploit tradeoff, with higher $\alpha$ corresponding to more exploration, and lower $\alpha$ corresponding to more exploitation. The right coefficient (the one which leads to the stablest / highest-reward learning) may vary from environment to environment, and could require careful tuning.

<div align="center">

![SAC]({{ site.baseurl }}/images/rl_SAC.png)

*Figure: Soft Actor-Critic (SAC) Algorithm Overview*

</div>

To train our RL agent, we employed the Soft Actor-Critic (SAC) algorithm. SAC is an off-policy, actor-critic reinforcement learning method, as illustrated in Figure above. Being off-policy means that the algorithm can learn not only from the actions taken by its current policy but also from experiences collected using different or past policies, which are stored in a replay buffer. This allows SAC to reuse data more efficiently compared to on-policy methods, making the training process more sample-efficient and stable.

**Actor–Critic Components in Reinforcement Learning:**


- **Actor**:  Chooses the best action $a_{t}$ to execute based on the current state $s_{t}$ of the environment. It represents the policy $\pi (a | s)$, which maps states to actions.

- **Critic (Q-function)**: Evaluates the action chosen by the actor by estimating the expected return (value) from taking action $a_{t}$ in state $s_{t}$. It essentially tells the actor **how good this choice was** by predicting the long-term reward.

- **State ($s_{t}$)**: The observation of the environment at a given time (e.g., robot position, sensor readings).

- **Action ($a_{t}$)**: The decision made by the actor (e.g., move forward, turn left, adjust speed).

- **Reward ($r_{t}$)**: The immediate feedback from the environment after executing an action (e.g., positive reward for reaching closer to the target, negative reward for hitting an obstacle).

- **Environment**: The world in which the agent operates. It provides new states and rewards in response to the agent’s actions, forming the continuous loop of interaction.

---

**Interaction Loop** as shown in **Figure above**:

1. The agent observes the current **state** $s_{t}$.  
2. The **Actor** selects an **action** $a_t$ using the the policy $\pi (a | s)$.  
3. The **Environment** responds with a new **state** $s_{t+1}$ and a **reward** $r_{t}$.  
4. The **Critic** evaluates how good the action was by estimating the value (Q-function).  
5. The Actor updates its policy to improve future decisions.

***Soft Actor-Critic (SAC)***: It uses a replay buffer to store past experiences, and allowing the model to learn from this data improving sample efficiency and training speed.

**Pseudocode:**

<div align="center">

![Pseudocode](/images/rl_pscode.png)

</div>

- I used the **Soft Actor-Critic (SAC)** implementation from the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) framework. This library provided the core training setup, including policy and value networks, replay buffer, and entropy regularization.  
- You can find the detailed documentation and examples [here](https://stable-baselines3.readthedocs.io/) for parameter fine-tuning and usage of the available functions.

## 2.2. Environment Setup

### 2.2.1. Frameworks

In reinforcement learning (RL), setting up the environment is crucial as it defines the interactions between the agent and its surroundings. The two main components used in this setup are the Gymnasium framework and the Stable-Baselines3 library, each serving a specific purpose to facilitate the RL training process. 

- Gymnasium framework (an updated version of OpenAI Gym): is a framework that standardizes the creation, manipulation, and interaction of RL environments. This framework allows for a consistent interface across various types of environments, making it easier to train, test, and benchmark RL algorithms across different setups. 
- Stable-Baselines3 library: is a popular Python library for implementing and managing reinforcement learning algorithms. It provides robust and optimized implementations of many commonly used RL algorithms, including Soft Actor-Critic (SAC), which is particularly well-suited for continuous control tasks like robotic movement.

<div align="center">

![rl_gym](/images/rl_gym.png)

*Figure: Reinforcement Learning Framework*

</div>

### 2.2.2. Gazebo world configuration

To train the reinforcement learning model effectively, it is essential to configure the environment settings within the Gazebo simulation. This involves defining various parameters in the sdf (Simulation Description Format) file, which describes the world in which the robot will operate.
- The sdf file is a powerful tool for creating realistic and complex environments, enabling you to
simulate various physical attributes, environmental conditions, and world properties.

<h2 align="center">Simulation Parameters in SDF file</h2>

<div align="center">

<table border="1" cellspacing="0" cellpadding="5">
  <tr>
    <th>Parameters</th>
    <th>Type</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Gravity</td>
    <td></td>
    <td>9.8 (m/s^2)</td>
  </tr>
  <tr>
    <td>max_step_size</td>
    <td></td>
    <td>0.001 (s)</td>
  </tr>
  <tr>
    <td>real_time_factor</td>
    <td></td>
    <td>1</td>
  </tr>
  <tr>
    <td>sensor_noise</td>
    <td></td>
    <td>“Gaussian”</td>
  </tr>
  <tr>
    <td>Friction coefficient (mu/mu2)</td>
    <td></td>
    <td>0.9 / 0.9</td>
  </tr>
  <tr>
    <td>Contact Dynamic (Kp, Kd)</td>
    <td></td>
    <td>100000 / 1.0</td>
  </tr>
</table>

</div>

- Real time factor (RTF) is the ratio of simulation time to real-world time. It measures how fast the simulation is running compared to real time.
- Max step size defines the time interval of each simulation step, essentially controlling the granularity of the physics calculations.
    - Smaller Step Size, more accurate and realistic simulation results.
    - Larger Step Size, faster simulation with less detail and accuracy.
- $\mu$ and $\mu_2$ allow for more precise control of frictional forces, enabling simulations of surfaces with different frictional behaviors depending on the direction of force application.
*Example in gazebo:* When configuring friction for a robot’s wheels, you might set different values for µ and µ2 to simulate the grip behavior of the tire differently in forward
and sideways directions. Common values:
    - 0.1 - 0.3: Low friction (e.g., ice or wet surfaces).
    - 0.5 - 0.8: Medium friction (e.g., wood or regular flooring).
    - 0.9 - 1.2: High friction (e.g., rubber on concrete or asphalt).
- kp (spring stiffness): A value of 100000.0 means the contact points between surfaces
(like robot wheels or feet touching the ground) will resist penetration with a high force,
simulating rigid surfaces.
- kd (damping coefficient): A low value like 1.0 means the system does not over-damp col-
lisions, allowing for some realistic bouncing or quick response without the robot getting
”stuck” or too slow after collisions.

### 2.2.3. Robot Configuration
In order to accurately simulate the behavior and capabilities of a real robot in Gazebo, it’s essential to configure the robot model to match the real robot’s kinematic parameters and sensor  specifications. This involves customizing several parameters in the SDF (Simulation Description Format) file or URDF (Unified Robot Description Format) file. Here’s an overview of key aspects to consider:

<h2 align="center">Physical Properties</h2>

<div align="center">

<table border="1" cellspacing="0" cellpadding="5">
  <tr>
    <th>Physical Properties</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>wheel separation</td>
    <td>0.28 (m)</td>
  </tr>
  <tr>
    <td>wheel diameter</td>
    <td>0.115 (m)</td>
  </tr>
  <tr>
    <td>max_wheel_torque</td>
    <td>5 (N/m) → approximate</td>
  </tr>
  <tr>
    <td>wheel acceleration</td>
    <td>0.6 (m/s^2) → approximate</td>
  </tr>
</table>

</div>

To accurately simulate the Laser (RPLidar A1) sensor in Gazebo world, you need to configure the sensor’s
parameters in the **SDF (Simulation Description Format)** file for the robot. These parameters
should closely match the real RPLidar A1’s specifications to achieve realistic sensing and perception.

<h2 align="center">Lidar Parameters</h2>

<div align="center">

<table border="1" cellspacing="0" cellpadding="5">
  <tr>
    <th>Parameters</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Sample</td>
    <td>360 (degree)</td>
  </tr>
  <tr>
    <td>update rate</td>
    <td>10 Hz</td>
  </tr>
  <tr>
    <td>Resolution</td>
    <td>0.017</td>
  </tr>
  <tr>
    <td>max_angle</td>
    <td>6.28 (2Pi)</td>
  </tr>
  <tr>
    <td>max_range</td>
    <td>12</td>
  </tr>
</table>

</div>

- Sample: This typically refers to the number of measurements or data points taken by the LiDAR sensor within a specific time frame or a single scan.
- Update Rate: This is the frequency at which the LiDAR sensor updates its data. It is usually measured in Hertz (Hz), meaning the number of times per second the sensor
captures and sends a new scan or data point.
- Resolution: The resolution of a LiDAR sensor refers to the level of detail in the measurements or the density of points in a scan.
- Max/Min Angle: These parameters define the angular range over which the LiDAR sensor scans.

## 2.3. Model Training
To train RL agent, we follow the several steps:
- Setup scenario for RL agent
- Reward System Design
- SAC Parameter Initialization and Type of Policy
### 2.3.1. Setup scenario for model training
We train the Reinforcement Learning (RL) model in a Gazebo simulation environment designed to mimic realistic navigation tasks. As shown in ***Figure below***, the setup includes:
- Random waypoints
- Fix robot Position
- Static obstacles
- A waypoint-based layout: The robot is required to navigate along three predefined paths, consist of 10 points.
- Reward system:
    - Reach target => reward + 10
    - Hit Obstacle => reward - 5

<div align="center">

![RL Way-Points](/images/rl_waypoint.png)

*Figure: Way-points Construction for 3 randoms path*

</div>

### 2.3.2. SAC Parameter Initialization
We utilize the Soft Actor-Critic (SAC) algorithm from the **“Stable Baselines3”** library to train our reinforcement learning (RL) model. The selected training parameters are outlined in the table below.

<h2 align="center">SAC Algorithm Parameters</h2>
<div align="center">
<table border="1" cellspacing="0" cellpadding="5">
  <tr>
    <th>Parameters</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Type Policy</td>
    <td>"MultiInputPolicy"</td>
  </tr>
  <tr>
    <td>verbose</td>
    <td>1</td>
  </tr>
  <tr>
    <td>learning_rate</td>
    <td>0.0003</td>
  </tr>
  <tr>
    <td>buffer_size</td>
    <td>1 000 000</td>
  </tr>
  <tr>
    <td>learning_starts</td>
    <td>100</td>
  </tr>
  <tr>
    <td>batch_size</td>
    <td>256</td>
  </tr>
  <tr>
    <td>tau</td>
    <td>0.05</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td>train_freq</td>
    <td>10</td>
  </tr>
  <tr>
    <td>device</td>
    <td>'auto'</td>
  </tr>
</table>
</div>

---

# III. Results and Discussion
In this section, we divide into 2 part of experiment:
- **Simulation**: Train the RL robot in the Gazebo world, which creates the virtual environment and configures it to closely match real-world testing conditions and sensor specifications.
- **Hardware**: Utilize a zero-shot Sim-to-Real approach by directly deploying the model to the hardware **(Raspberry Pi 4B)** via a local network, without any fine-tuning.

## 3.1. Simulation
There are 2 models that we use to conduct the experiment in the simulation, “SAC_waypoint02” and “SAC_waypoint03”.

### 3.1.1. Model “SAC_waypoint02”
The model trains around ”3 million timesteps” with “maximum episode 300”, Using an NVIDIA RTX 3050Ti GPU.

<div align="center">

![RL result 1](/images/rl_rs01.png)

*Figure: Evaluation Episode length and Reward*

</div>

