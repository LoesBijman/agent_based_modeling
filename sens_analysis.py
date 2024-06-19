from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector
from model import CrowdModel
import pandas as pd
import numpy as np
from tqdm import tqdm

def run_sensitivity_analysis(param_name, param_values, num_iterations=10):
    results = []
    default_params = {
        "width": 20,
        "height": 20,
        "N": 100,
        "goal_radius": 10,
        "knowledge_spread_prob": 0.5,
        "interaction_radius": 1,
        "exit_awareness_radius": 3,
        "disaster_knowledge_prop": 0.2,
        "environment_knowledge_prop": 0.2
    }

    for value in tqdm(param_values, desc=f"Running sensitivity analysis on {param_name}"):
        for iteration in range(num_iterations):
            params = default_params.copy()
            params[param_name] = value

            model = CrowdModel(**params)
            model.run_model()

            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()

            results.append({
                "parameter": param_name,
                "value": value,
                "iteration": iteration,
                "steps_to_completion": model.steps_to_completion,
                "total_agents": params["N"],
                "grid_size": params["width"] * params["height"],
                "goal_radius": params["goal_radius"],
                "knowledge_spread_prob": params["knowledge_spread_prob"],
                "interaction_radius": params["interaction_radius"],
                "exit_awareness_radius": params["exit_awareness_radius"],
                "disaster_knowledge_prop": params["disaster_knowledge_prop"],
                "environment_knowledge_prop": params["environment_knowledge_prop"],
                "model_data": model_data.to_dict(),
                "agent_data": agent_data.to_dict()
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    knowledge_spread_prob_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    interaction_radius_values = [1, 2, 3, 4, 5]
    exit_awareness_radius_values = [2, 3, 4, 5, 6]
    disaster_knowledge_prop_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    environment_knowledge_prop_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    results_knowledge_spread = run_sensitivity_analysis("knowledge_spread_prob", knowledge_spread_prob_values)
    results_interaction_radius = run_sensitivity_analysis("interaction_radius", interaction_radius_values)
    results_exit_awareness = run_sensitivity_analysis("exit_awareness_radius", exit_awareness_radius_values)
    results_disaster_knowledge = run_sensitivity_analysis("disaster_knowledge_prop", disaster_knowledge_prop_values)
    results_environment_knowledge = run_sensitivity_analysis("environment_knowledge_prop", environment_knowledge_prop_values)

    all_results = pd.concat([
        results_knowledge_spread,
        results_interaction_radius,
        results_exit_awareness,
        results_disaster_knowledge,
        results_environment_knowledge
    ], ignore_index=True)

    all_results.to_csv("sensitivity_analysis_results.csv", index=False)
