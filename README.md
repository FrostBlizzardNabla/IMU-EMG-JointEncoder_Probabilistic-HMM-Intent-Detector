# Sensor-Fusion Elbow Joint Simulator

A real-time sensor-fusion simulation of elbow joint motion using **EMG, IMU, and encoder data**, combined through a **Hidden Markov Model (HMM)** to perform continuous **intent inference** and **biomechanical visualization**.

## Overview

This project models a single-cycle elbow movement (flexion → extension → flexion) and fuses multimodal sensor inputs to estimate the **probability of joint activation and movement intent** over time.

The system integrates:

* **EMG signals** (biceps & triceps) for muscle activation
* **Encoder data** for joint angle tracking
* **IMU data** for motion dynamics

These signals are processed through an HMM to infer whether the user intends to:

* Flex the elbow
* Extend the elbow
* Remain at rest

## Features

* Real-time **link-based biomechanical visualization** of the elbow joint
* Interactive **slider-controlled motion cycle exploration**
* Continuous **intent probability estimation** (not discrete states)
* Multi-sensor overlay:

  * EMG (biceps/triceps)
  * Joint angle (encoder)
  * IMU motion data
* HMM-based **state transition modeling** for realistic motion inference

## Intent Modeling Logic

The system defines intent as probabilistic activation derived from sensor agreement:

* **Flexion Intent**
  High when:

  * Biceps EMG ↑
  * Joint angle increasing
  * IMU indicates inward motion

* **Extension Intent**
  High when:

  * Triceps EMG ↑
  * Joint angle decreasing
  * IMU indicates outward motion

This results in a continuous probability curve representing **likelihood of joint activation**, rather than a simple threshold or binary classification.

## Visualization

The interface includes:

* A **link diagram** of the upper arm and forearm
* A **time-synchronized slider** to scrub through motion
* Live plots of:

  * EMG signals
  * Joint angle
  * IMU data
  * Intent probability distribution

## Applications

* Human–machine interaction research
* Exoskeleton and prosthetic control systems
* Biomechanics and motion analysis
* Sensor fusion algorithm development

## Future Work

* Extend to **multi-joint systems (shoulder + elbow)**
* Integrate real-time hardware input
* Upgrade to **deep sequential models (LSTM / Hybrid HMM)**
* Add adaptive calibration for individual users

---

Built as a foundation for **intuitive, biologically-informed control systems**.
