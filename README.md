
# Evaluating the Impact of Environmental Familiarity and Intervention Strategies on Crowd Evacuation Dynamics in Emergency Situations

### Esther Bakels 12362980, Koen Weverink 14711982, Loes Bijman 15211312, Guoda Paulauskaite 12255076, Sacha Gijsbers 12798525 

### _Agent-Based Modeling Course 2024 | MSc Computational Science (UvA/VU)_

## Introduction

This study uses agent-based modeling to investigate how environmental familiarity (knowledge of exits) and intervention strategies (such as disaster announcements, guides, and exit signage) affect crowd dynamics during emergency evacuations. The model developed in this GitHub repository seeks to answer the following research question: _How do varying levels of environmental familiarity and the implementation of intervention strategies influence the speed of crowd evacuations in public spaces under an emergency situation?_ 

Through simulations of this model, this study aims to provide insights that inform the design of more effective evacuation protocols tailored to diverse environmental and social contexts.

## Table of Contents

1. [Features](#features)
2. [Simulation Components](#simulation-components)
3. [How It Works](#how-it-works)
4. [Prerequisites](#prerequisites)
5. [Clone the Repo](#clone-the-repo)
6. [Running the Simulation](#running-the-simulation)
7. [Experiments](#experiments)
8. [Local and Global Sensitivity Analysis](#local-and-global-sensitivity-analysis)
9. [Stats](#stats)

## Features

- **Knowledge Acquisition:** Agents learn about disasters and exits within a certain radius.
- **Knowledge Spreading:** Agents spread knowledge about disasters to other agents within a social radius.
- **Evacuator Influence:** Presence of an evacuator agent enhances knowledge spreading about disasters and exits.
- **Stochastic Knowledge Spreading:** The probability of knowledge spreading from one agent to another is determined stochastically.

## Simulation Components

- **Agents:** Represent individuals in the crowd with varying levels of knowledge about the environment and disaster.
- **Goals:** Represent exits or safe locations. Agents learn about these goals if they come within a certain radius.
- **Fire:** Represents the disaster. Agents learn about the fire if they come within a certain radius of it.
- **Evacuator:** A special agent that significantly increases the knowledge spreading about the disaster and exits.

## How It Works

1. **Learning about Goals and Fire:** At each step, agents check their surroundings for goals (exits) and the fire. If these are within a certain radius, the agent updates its knowledge.
2. **Spreading Knowledge:** Agents with knowledge of the disaster spread this knowledge to nearby agents. The probability of spreading knowledge is determined by a predefined probability.
3. **Evacuator Presence:** If an evacuator is present and within a certain radius of an agent, the agent immediately gains knowledge of the disaster and all exits.

## Prerequisites

The following Python packages are needed to run the model:

- Mesa (version 2.3.0)
- Numpy
- SciPy
- Pandas
- Matplotlib
- SAlib (version 1.5.0)

You can install the required packages using the following command:

```bash
pip install mesa numpy scipy pandas matplotlib salib
```

## Clone the Repo

Clone the repository to your local machine:
```bash
git clone https://github.com/LoesBijman/agent_based_modeling.git
cd agent_based_modeling
```

## Running the Simulation

To run the simulation, execute the `social_force_model.py` script in your Python environment:

```bash
python social_force_model.py
```

## Experiments

### Important Note!

Before running the experiment file, please comment out the following line (613) from _social_force_model.py_:

```python
server.launch()
```

### Initial Setup and Configuration

The experiments to investigate how environmental familiarity and intervention strategies affect crowd dynamics during emergency evacuations were done in the _sf_experiments.ipynb_ file in the _social_force_model_ folder. The results are plotted and saved in the _experiment_plots_ folder. The notebook includes setting up parameters for the crowd model simulation, such as grid dimensions, number of agents, fire radius, social radius, and probabilities related to spreading information.

### Running the Experiments

To run all the experiments, simply press _Run All_ in the notebook. Please note that this will take a long time.

### Visualization (Optional)

Code snippets are included to visualize the grid and the simulation process if needed. This part is commented out. Uncomment if visual inspection is wanted.

### Intervention 1 - Basic Model

The basic model is set up without any additional intervention. It runs the simulation, collects data on agents removed, agents who know about the fire, knowledge spread, goal changes, and clustering coefficients. Data is saved to CSV files.

### Intervention 2 - Fire Announcement

In this intervention, a fire announcement is made to see how it affects the evacuation process. Similar to the basic model, it runs multiple simulations, collects data, and saves it to CSV files.

### Intervention 3 - Evacuator

This intervention introduces an evacuator with an increased social radius. The model runs, and data is collected in the same manner as the previous interventions.

### Data Loading and Analysis

CSV files generated from the interventions are loaded for analysis. This includes calculating and plotting the clustering coefficients for different interventions and performing statistical tests to compare them.

### Visualization of Clustering Coefficients

The clustering coefficients for different interventions are plotted, showing the mean and standard deviation over time. Labels and legends help differentiate between the interventions.

### Statistical Testing

T-tests are performed to compare the mean clustering coefficients of the basic model against each intervention to determine if there are significant differences.

## Local and Global Sensitivity Analysis

### Configuration and Setup

The LSA is done in the _sensitivity.ipynb_. The GSA in _sobol_analysis.ipynb_. They both function similarly.

Logging is configured for tracking the progress of the analysis. A function is defined to calculate a scaling factor based on the grid dimension. The analysis problem is defined with 9 parameters, each with specific bounds:

- social_radius
- fire_radius
- exit_radius
- p_spreading
- p_spreading_environment
- gumbel_plus_mean
- gumbel_plus_std
- gumbel_minus_mean
- gumbel_minus_std

Fixed parameters for the model are set, including grid dimensions, number of agents, knowledge parameters, and fire avoidance radius.

### Model Initialization

A function to initialize and run the crowd model with given parameters and sample values from the Sobol analysis is defined. Another function to adjust exit locations based on radius is defined.

### Running the Sobol Sensitivity Analysis

The main function, `run_sobol_sensitivity_analysis`, uses the defined Sobol problem and model to run the sensitivity analysis over multiple iterations, saving results at specified checkpoints. The final results are saved to a CSV file.

### Executing the Analysis

The Sobol sensitivity analysis is executed if the script is run directly, not when imported as a module. The results are saved in a CSV file named "sobol_sensitivity_analysis_results.csv". The Sobol sensitivity analysis results are printed, showing the first order, total order, and second order indices. A bar plot is created to visualize the sensitivity indices with error bars, displaying the importance of each parameter. The plot is saved as "sobol_sensitivity_analysis_results.png". The plot is displayed within the notebook.

## Stats

### Data Loading

CSV files are loaded for different scenarios: basic, signage, fire_announcement, and evacuator for metrics such as agents removed, agents who know about the fire, exit knowledge spread, and goal changes.

### Statistical Tests

**Agents Removed:**

A t-test is performed to compare the mean number of agents removed between the basic scenario and each intervention (signage, fire_announcement, evacuator). The p-values are printed for each comparison.

**Knowledge of Disaster:**

Similar t-tests are performed to compare the mean number of agents who have knowledge of the disaster across scenarios.

**Spreading Exit Knowledge:**

T-tests are used to compare the mean spread of exit knowledge between the basic scenario and interventions.

**Changing Goals:**

T-tests are conducted to compare mean changes in goals across different scenarios.

### Tukey's HSD Test

Data is concatenated for each metric (agents removed, knowledge of disaster, exit knowledge spread, changing goals). Tukey's Honest Significant Difference (HSD) test is performed to make multiple comparisons between the groups (scenarios) for each metric. Results and summaries of Tukey's HSD tests are printed for each metric.
