from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np
import pandas as pd

from model_loes import CrowdModel, portrayal

# Initialize visualization
grid = CanvasGrid(portrayal, 20, 20, 500, 500)

width = 20
height = 20
exits = [ {"location": (0, height - 1), "radius": width // 2},
          {"location": (width - 1, 0), "radius": width // 2},
          {"location": (width - 1, height - 1), "radius": width // 2}]

# Initialize the server with the model
server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 20, "height": 20, "N": 100, "fire_radius": 20//3, 'social_radius': 20//4, 'p_spreading': 0.2, 'p_spreading_environment': 0.3, 'exits': exits})
server.model = CrowdModel(20, 20, 100, 20//3, 20//4, 0.2, 0.3, exits)  # Reset the model
server.model.run_model()
data = server.model.datacollector.get_model_vars_dataframe()

# Save each metric to a separate CSV file
data_agents_removed = data["Agents Removed"]
data_agents_know_fire = data["Agents Know Fire"]
data_exit_knowledge_spread = data["Exit Knowledge Spread"]
data_change_goal = data["Change Goal"]

data_agents_removed.to_csv("agents_removed_per_step.csv", index=False)
data_agents_know_fire.to_csv("agents_know_fire_per_step.csv", index=False)
data_exit_knowledge_spread.to_csv("exit_knowledge_spread_per_step.csv", index=False)
data_change_goal.to_csv("change_goal_per_step.csv", index=False)
# data.to_csv("agents_removed_per_step.csv", index=False)

print("Data 1 saved successfully!")

# Define the number of runs
num_runs = 5  # Adjust as needed

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Run the model multiple times
# for i in range(num_runs):
#     print(f"Running model {i + 1}...")
#     server.model = CrowdModel(20, 20, 100, 20//3, 20//4, 0.2, 0.3, exits)  # Reset the model
#     server.model.run_model()  # Run the model
    
#     # Get data for the current run
#     data_new = server.model.datacollector.get_model_vars_dataframe()
#     data_new = data_new["Agents Removed"]
    
#     # Rename columns to include run number
#     data_new.columns = [f"Run_{i+1}_{col}" for col in data_new.columns]
    
#     data_new = data_new.astype(int)
    
#     # Concatenate data for the current run to all_data
#     all_data = pd.concat([all_data, data_new], axis=1)

# # Save all_data to CSV with each run's data in separate columns
# all_data.to_csv("agents_removed_per_step_all_runs.csv", index=False)

# print("Data 2 saved successfully!")