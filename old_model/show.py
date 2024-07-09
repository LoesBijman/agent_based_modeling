import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
import pandas as pd

df = pd.read_csv('sensitivity_analysis_results.csv')

# Define the problem
problem = {
    'num_vars': 3,
    'names': ['social_radius', 'fire_radius', 'p_spreading'],
    'bounds': [
        [df['social_radius'].min(), df['social_radius'].max()],
        [df['fire_radius'].min(), df['fire_radius'].max()],
        [df['p_spreading'].min(), df['p_spreading'].max()]
    ]
}

# Determine the maximum number of samples based on the available data
max_samples = 2**int(np.log2(len(df)))

# Generate samples
param_values = sobol_sample.sample(problem, max_samples)

# Evaluate the model (assuming the output we are interested in is 'Agents Removed')
Y = df['Agents Removed'].values[:len(param_values)]

# Perform Sobol sensitivity analysis
sobol_indices = sobol.analyze(problem, Y, print_to_console=False)

# Extract indices
S1 = sobol_indices['S1']
ST = sobol_indices['ST']
S2 = sobol_indices['S2']

# Prepare data for DataFrame
S2_df = pd.DataFrame(S2, index=problem['names'], columns=problem['names'])

# Replace NaN values with a placeholder (e.g., 0) for better readability
S2_df.fillna(0, inplace=True)

# Combine S1 and ST indices
sobol_df = pd.DataFrame({
    'S1': S1,
    'ST': ST
}, index=problem['names'])

# Display the results
print("Sobol Sensitivity Analysis Results (S1 and ST):")
print(sobol_df)

print("\nSobol Sensitivity Analysis Results (S2):")
print(S2_df)

# Save the results to CSV files
sobol_df.to_csv('sobol_sensitivity_analysis_S1_ST.csv')
S2_df.to_csv('sobol_sensitivity_analysis_S2.csv')

print("Results saved to 'sobol_sensitivity_analysis_S1_ST.csv' and 'sobol_sensitivity_analysis_S2.csv'")
