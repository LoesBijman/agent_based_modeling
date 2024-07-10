# Evaluating the Impact of Environmental Familiarity and Intervention Strategies on Crowd Evacuation Dynamics in Emergency Situations
### _Agent-Based Modelling Course 2024 | MSc Computational Science (UvA/VU)_

This study uses agent-based modelling to investigate how environmental familiarity (knowledge of exits)
and intervention strategies (such as disaster announcement, guides and exit signage) affect 
crowd dynamics during emergency evacuations. The model build in this Github repository seeks to answer the following research question: _How do varying levels of environmental familiarity and the implementation of intervention strategies influence the speed of crowd evacuations in public spaces under an emergency situation?_ Through computational modeling and simulations, this research aims to provide insights that inform the design of more effective evacuation protocols tailored to diverse environmental and social contexts.

### Prerequisites
The following Python packages are needed to run the model:
1. **Mesa** (version 2.3.0)
2. **Numpy**
3. **SciPy**
4. **Pandas**
5. **Matplotlib**
6. **SAlib** (version 1.5.0)


### Visualizing the model
The model was constructed using the Mesa library and can be run and visualized by executing _social_force_model.py_ in the _social_force_model_ folder. The parameters can be changed in this file as needed. 
The experiments done to investigate how environmental familiarity and and intervention strategies affect 
crowd dynamics during emergency evacuations were done in the _sf_experiments.ipynb_ file in the _social_force_model_ folder. The results are plotted and saved in the _experiment_plots_ folder. 
