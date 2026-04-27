# Deep-Q Cart-Pole Visualization

## Project Summary
This project implements a reinforcement learning visualization of the Cart-Pole problem using the Model-View-Controller (MVC) architecture. Its trained via Proximal Policy Optimization (PPO) and has an interactive Pygame interface. Users can configure physical parameters such as pole length and mass and apply manual disturbances (nudges) to test the models stability and recovery logic.

## Project Structure
* **main.py**: This manages the program loop intializing the different classes and calling relevant functions
* **model.py**: Contains the CustomCartPole class. This module defines the physics environment and the reward function used for agent stability.
* **view.py**: Handles all Pygame rendering logic, including the HUD, sensor bars, and user input boxes.
* **controller.py**: Handles the interaction between the Model and View, managing user inputs and the model predictions.

## Installation and Requirements

### Dependencies
This project requires the following Python libraries:
* pygame-ce
* numpy
* gymnasium
* stable-baselines3

### Setup Instructions
To run this project on your machine, follow these steps:

1. Clone the repository to your local directory.
2. Install the required dependencies using the provided requirements file:
   ```bash
   pip install -r requirements.txt