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
# server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 20, "height": 20, "N": 100, "goal_radius": 10, "fire_radius": 10, "fire_locations": [[0,0], [0,1], [0,2]], 'social_radius': 2})
server.port = 9999
# server.launch()
data = server.model.datacollector.get_model_vars_dataframe()
data.to_csv("agents_removed_per_step.csv", index=False)

print("Data 1 saved successfully!")

# Define the number of runs
num_runs = 5  # Adjust as needed

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Run the model multiple times
# for i in range(num_runs):
#     print(f"Running model {i + 1}...")
#     server.model = CrowdModel(20, 20, 100, 10, 10, [[0,0], [0,1], [0,2]], 2)  # Reset the model
#     server.model.run_model()  # Run the model
    
#     # Get data for the current run
#     data_new = server.model.datacollector.get_model_vars_dataframe()
    
#     # Rename columns to include run number
#     data_new.columns = [f"Run_{i+1}_{col}" for col in data_new.columns]
    
#     data_new = data_new.astype(int)
    
#     # Concatenate data for the current run to all_data
#     all_data = pd.concat([all_data, data_new], axis=1)

# # Save all_data to CSV with each run's data in separate columns
# all_data.to_csv("agents_removed_per_step_all_runs.csv", index=False)

# print("Data 2 saved successfully!")