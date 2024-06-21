from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector

from tqdm import tqdm

from model import CrowdModel
from mesa.batchrunner import batch_run
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Define the parameter bounds for sensitivity analysis
parameter_bounds = {
    "social_radius": range(0, 15),  # Vary social radius from 2 to 5
    "fire_radius": range(0, 15, 2),  # Vary fire radius from 5 to 10
    "exit_radius": [0, 5, 10, 15, 20],  # Vary exit radius from 5 to 15
    "p_spreading": np.arange(0, 1, 0.2),  # Vary spreading probability from 0.1 to 0.5
    "p_spreading_environment": np.arange(0, 1, 0.2)  # Vary environment knowledge spreading probability from 0.1 to 0.5
}

# Define fixed parameters
fixed_params = {
    "width": 25,
    "height": 25,
    "N": int(0.25 * 25 * 25),  # 25% of the grid size
    "p_env_knowledge_params": [0,1,-1,1]
}

# Function to adjust exits based on radius
def get_exits_with_radius(radius):
    return [
        {"location": (0, 24), "radius": radius},
        {"location": (24, 0), "radius": radius},
        {"location": (24, 24), "radius": radius}
    ]

# Run the sensitivity analysis for each parameter
def run_sensitivity_analysis():
    all_results = []
    
    for param, values in parameter_bounds.items():
        print(f"Running simulations for parameter: {param}")
        
        for value in values:
            # Create the parameter set for the current run
            param_set = {
                "social_radius": 4,  # Default value, can be adjusted
                "fire_radius": 7,    # Default value, can be adjusted
                "exits": get_exits_with_radius(10),  # Default value, can be adjusted
                "p_spreading": 0.3,  # Default value, can be adjusted
                "p_spreading_environment": 0.3  # Default value, can be adjusted
            }
            
            # Update the parameter set with the current parameter and value
            if param == "exit_radius":
                param_set["exits"] = get_exits_with_radius(value)
            else:
                param_set[param] = value
            
            # Combine with fixed parameters
            param_set.update(fixed_params)
            
            # Convert param_set to a format suitable for batch_run
            param_dict = {k: [v] for k, v in param_set.items()}
            
            # Run the batch
            try:
                results = batch_run(
                    model_cls=CrowdModel,
                    parameters=param_dict,
                    number_processes=None,  # Use all available processors
                    iterations=10,  # Number of iterations per parameter set
                    max_steps=700,  # Max steps for each run
                    data_collection_period=-1,  # Collect data at each step
                    display_progress=True  # Display progress bar
                )
            except Exception as e:
                print(f"Error during batch_run for {param} = {value}: {e}")
                continue

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results)
            
            # Include the parameter name and value in the results
            results_df['parameter'] = param
            results_df['value'] = value
            
            # Append to all results
            all_results.append(results_df)
    
    # Combine all results into a single DataFrame
    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results

# Ensure this code runs only when executed as a script, not when imported as a module
if __name__ == "__main__":
    # Run the simulation and collect data
    results_df = run_sensitivity_analysis()

    # Save the results to a CSV file
    results_df.to_csv("sensitivity_analysis_results.csv", index=False)

    print("Sensitivity analysis completed and results saved to sensitivity_analysis_results.csv!")