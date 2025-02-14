from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np
from scipy.stats import gumbel_r


class CrowdAgent(Agent):
    """
    Represents an agent in a crowd simulation model.

    Attributes:
        unique_id (int): The unique identifier of the agent.
        model (object): The model that the agent belongs to.
        goals (list): A list of goals that the agent can move towards.
        current_goal (dict): The current goal that the agent is moving towards.

    Methods:
        step(): Performs a step in the agent's behavior.
        move_towards_goal(): Moves the agent towards its current goal.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.current_goal = None
        self.model.goals

        self.knowledge_of_disaster = False

        #set environment knowledge stochastically
        # env_knowledge_chance = gumbel_r.rvs(loc = self.model.p_env_knowledge_params[0], scale = self.model.p_env_knowledge_params[1], size = 1)
        env_knowledge_chance = np.random.uniform(0,1)
        self.knowledge_of_environment = []

        if env_knowledge_chance < self.model.p_env_knowledge_params[0]:
            pass
        elif env_knowledge_chance < self.model.p_env_knowledge_params[1]:
            self.knowledge_of_environment = [self.model.goals[0]['location']]
        else:
            for goal_dict in self.model.goals:
                self.knowledge_of_environment.append(goal_dict['location'])


    def step(self):
        """
        Performs a step in the agent's behavior.
        """
        
        # Update the agent's knowledge of the environment according to radius of exit
        for goal in self.model.goals:
            if goal['location'] not in self.knowledge_of_environment:
                if np.linalg.norm(np.array(self.pos) - np.array(goal['location'])) < goal['radius']:
                    if not self.knowledge_of_environment:
                        self.model.num_agents_know_an_exit += 1 ############# ADDING    
                    self.knowledge_of_environment.append(goal['location'])   

        # Spread knowledge of the environment
        self.spread_knowledge()
        
        # Update the agent's knowledge of the disaster
        for fire in self.model.fire:
            if np.linalg.norm(np.array(self.pos) - np.array(fire.pos)) < self.model.fire_radius:
                if not self.knowledge_of_disaster:
                    self.knowledge_of_disaster = True
                    self.model.num_agents_know_fire += 1 # Count
        
        disaster_knowing_agents = [agent for agent in self.model.schedule.agents if isinstance(agent, CrowdAgent) and agent.knowledge_of_disaster]

        #If evacuator present, the agent gains knowledge of disaster and all exits
        if self.model.evacuator_present:
            if np.linalg.norm(np.array(self.pos) - np.array(self.model.evacuator[0].pos)) < self.model.evacuator_radius:
                if not self.knowledge_of_disaster:
                    self.knowledge_of_disaster = True
                    self.model.num_agents_know_fire += 1 # Count

                current_knowledge = self.knowledge_of_environment
            
                for goal_dict in self.model.goals: #add exit knowledge
                    if goal_dict not in current_knowledge:
                        if not self.knowledge_of_environment:
                            self.model.num_agents_know_an_exit += 1 ############# ADDING  
                        self.knowledge_of_environment.append(goal_dict['location'])  
                
        # Perform step
        if not self.knowledge_of_disaster:
            for dis_agent in disaster_knowing_agents:
                if np.linalg.norm(np.array(self.pos) - np.array(dis_agent.pos)) < self.model.social_radius:
                    if not self.knowledge_of_disaster:
                        # spread knowledge stochastically
                        u = np.random.uniform(0,1)
                        if u < self.model.p_spreading:
                            self.knowledge_of_disaster = True
                            self.model.num_agents_know_fire += 1 # Count
            self.stand_still()

        else:
            # check if neigbor is fire
            fire = self.check_fire(self.pos)
            if fire:
                # move away from fire to oposite direction
                x, y = self.pos
                x_fire, y_fire = fire
                if x_fire > x:
                    x -= 1
                elif x_fire < x:
                    x += 1
                if y_fire > y:
                    y -= 1
                elif y_fire < y:
                    y += 1
                
                # Ensure the new position is within grid bounds and empty
                if 0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height and self.model.grid.is_cell_empty((x, y)):
                    self.model.grid.move_agent(self, (x, y))   
            elif self.knowledge_of_environment:
                # print current goal
                goals_of_agents = self.knowledge_of_environment

                # Calculate the distances
                distances = np.linalg.norm(np.array(goals_of_agents) - np.array(self.pos), axis=1)

                # Find the index of the minimum distance
                min_index = np.argmin(distances)

                # Get the coordinates that are closest to your coordinates
                closest_coords = goals_of_agents[min_index]

                if self.current_goal != closest_coords and self.current_goal is not None:
                    self.model.change_goal += 1 # Count
                    
                self.current_goal = closest_coords 
                self.move_towards_goal()
            else:
                # random exploration
                self.random_movement()
    
    def check_fire(self, pos):
        for fire in self.model.fire:
            if np.linalg.norm(np.array(pos) - np.array(fire.pos)) <= 1:
                return fire.pos
        return False
                        
    def spread_knowledge(self):
        if self.random.random() < self.model.p_spreading_environment:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=self.model.social_radius)
            for neighbor in neighbors:
                if isinstance(neighbor, CrowdAgent):
                    if neighbor.current_goal:
                        if neighbor.current_goal not in self.knowledge_of_environment:
                            if not self.knowledge_of_environment:
                                self.model.num_agents_know_an_exit += 1 ############# ADDING 
                            self.knowledge_of_environment.append(neighbor.current_goal)
                            self.model.exit_knowledge_spread += 1 # Count
                    
    def move_towards_goal(self):
        """
        Moves the agent towards its current goal.
        """
        if self.current_goal is None:
            return  # No goal to move towards

        # Own position and goal position
        x, y = self.pos
        goal_x, goal_y = self.current_goal

        # Calculate the distance to the goal for all neighboring cells
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height]
        distances = [(neighbor, np.linalg.norm(np.array(neighbor) - np.array(self.current_goal)))
                     for neighbor in valid_neighbors if self.model.grid.is_cell_empty(neighbor) or neighbor == self.current_goal]
        
        # Move to the neighboring cell that is closest to the goal and empty    
        if distances:
            near_fire = True
            while near_fire and distances:
                # Get the neighbor closest to the goal
                next_move, _ = min(distances, key=lambda t: t[1])
                
                near_fire = False
                for fire in self.model.fire:
                    if np.linalg.norm(np.array(next_move) - np.array(fire.pos)) <= 1:
                        near_fire = True
                        break
                
                # If the chosen move is near fire, remove it from the distances list
                if near_fire:
                    distances = [d for d in distances if d[0] != next_move]

            # Only move if we found a valid next move not near fire
            if not near_fire:
                self.model.grid.move_agent(self, next_move)
        
        # Check if the agent has reached the goal or adjacent cell, and remove it from the model if it has
        if self.pos == self.current_goal:
            # print(f"Agent {self.unique_id} reached the goal!")
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def stand_still(self):
        """
        The agent stands still.
        """
        pass  # Do nothing, agent stands still

    def random_movement(self):
        """
        The agent moves randomly.
        """
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        for step in possible_steps:
            for fire in self.model.fire:
                if np.linalg.norm(np.array(step) - np.array(fire.pos)) <= 1:
                    possible_steps = tuple([x for x in possible_steps if x != step])
                    break
        new_position = self.random.choice(possible_steps)
        if self.model.grid.is_cell_empty(new_position):
            self.model.grid.move_agent(self, new_position)


class CrowdModel(Model):
    """
    A class representing a crowd model.

    Attributes:
        width (int): The width of the model's grid.
        height (int): The height of the model's grid.
        N (int): The number of agents in the model.
        goal_radius (float): The radius of the goal area.

    Methods:
        __init__(self, width, height, N, goal_radius): Initializes the crowd model.
        step(self): Advances the model by one step.
    """

    def __init__(self, width, height, N, p_env_knowledge_params, fire_radius, social_radius, p_spreading, p_spreading_environment, exits, evacuator_present = False, evacuator_radius = None):
        """
        Args:
            width (int): The width of the model's grid.
            height (int): The height of the model's grid.
            N (int): The number of agents in the model.
        """

        self.N = N
        self.num_agents = N - 1 - len(exits)
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.fire_radius = fire_radius
        self.social_radius = social_radius
        self.p_spreading = p_spreading
        self.p_spreading_environment = p_spreading_environment
        self.p_env_knowledge_params = p_env_knowledge_params
        self.evacuator_present = evacuator_present
        self.evacuator_radius = evacuator_radius
        self.running = True  # Initialize the running state


        # self.datacollector = DataCollector(
        #     {"Agents Removed": lambda m: m.num_agents_removed}
        # )
        # self.num_agents_removed = 0  # Track the number of agents removed

        # Save data
        self.datacollector = DataCollector(
            {"Agents Removed": lambda m: m.num_agents_removed, 
             "Agents Know Fire": lambda m: m.num_agents_know_fire,
             "Exit Knowledge Spread": lambda m: m.exit_knowledge_spread,
             "Change Goal": lambda m: m.change_goal,
             "Exit Knowledge": lambda m: m.num_agents_know_an_exit})
        self.num_agents_removed = 0  # Number of agents removed
        self.num_agents_know_fire = 0 # Number of agents that know about the fire
        self.exit_knowledge_spread = 0 # Number of times agent tells another agent about a new exit
        self.change_goal = 0 # Number of times someone changes direction to a closer goal
        self.num_agents_know_an_exit = 0 # Number of agents that know about an exit

        self.goals = exits

        for i, exit in enumerate(exits):
            x, y = exit["location"]
            goal = Goal(i, self)
            self.schedule.add(goal)
            self.grid.place_agent(goal, (x,y))

        #Spawn an evacuator if in intervention mode
        if evacuator_present:
            evacuator = Evacuator(i, self)
            self.schedule.add(evacuator)
            self.grid.place_agent(evacuator, (width // 2, height // 2))
            self.evacuator = [agent for agent in self.schedule.agents if isinstance(agent, Evacuator)]

        # Create a fire
        x = int(np.round(np.random.uniform(2, width - 3)))
        y = int(np.round(np.random.uniform(2, height - 3)))
        if self.evacuator_present:
            #spawn the fire outside evacuator radius
            while np.linalg.norm(np.array([x, y]) - np.array(self.evacuator[0].pos)) < self.evacuator_radius: 
                x = int(np.round(np.random.uniform(2, width - 3)))
                y = int(np.round(np.random.uniform(2, height - 3))) 

        fire = Hazard(i, self)
        self.schedule.add(fire)
        self.grid.place_agent(fire, (x,y))

        # retrieve the fire locations
        self.fire = [agent for agent in self.schedule.agents if isinstance(agent, Hazard)]
            
        # Create agents and place them in the model
        for i in range(self.num_agents):
            x = self.random.randint(0, width - 1)
            y = self.random.randint(0, height - 1)
            while (x,y) in self.fire:
                x = self.random.randint(0, width - 1)
                y = self.random.randint(0, height - 1)
            agent = CrowdAgent(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y))
            # check if agent knows an exit
            if agent.knowledge_of_environment:
                print("self.schedule.steps", self.schedule.steps)
                self.num_agents_know_an_exit += 1


    def step(self):
        """
        Advances the model by one step.
        """
        # Collect data on the number of CrowdAgents removed
        
        self.num_agents_removed = self.num_agents - sum(1 for agent in self.schedule.agents if isinstance(agent, CrowdAgent))
        self.num_agents_know_fire = self.num_agents_know_fire
        self.exit_knowledge_spread = self.exit_knowledge_spread
        self.change_goal = self.change_goal
        self.num_agents_know_an_exit = self.num_agents_know_an_exit
        self.datacollector.collect(self)
        self.schedule.step()
        
        print(f"Step: {self.schedule.steps}")
        
        # if self.num_agents_removed == self.num_agents:
        #     self.running = False
        #     # print the number of steps it took for all agents to reach the goal
        #     print(f"Number of steps: {self.schedule.steps}")
        
        if self.schedule.steps == 150:
            self.running = False
            print(f"Number of steps: {self.schedule.steps}")
            
        # if all(not agent.knowledge_of_disaster for agent in self.schedule.agents if isinstance(agent, CrowdAgent)):
        #     self.running = False
        #     print(f"Evacuation failed... Number of people left: {self.num_agents - self.num_agents_removed}")
        #     print(f"Number of steps: {self.schedule.steps}")

class Hazard(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Goal(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Evacuator(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


def portrayal(agent):
    """
    Returns the visualization for a given agent.

    Parameters:
    agent (object): The agent for which the portrayal is being generated.

    Returns:
    dict: The portrayal dictionary containing the agent's shape, color, size, and layer.
    """
    if isinstance(agent, CrowdAgent):
        if agent.knowledge_of_disaster:
            if agent.knowledge_of_environment:
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "r": 0.5,
                    "Color": "green",
                    "Layer": 0
                }
            else:
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "r": 0.5,
                    "Color": "yellow",
                    "Layer": 0
                }
        else:
            portrayal = {
                "Shape": "circle",
                "Filled": "true",
                "r": 0.5,
                "Color": "blue",
                "Layer": 0
            }
    elif isinstance(agent, Hazard):
        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "r": 0.5,
            "Color": "red",
            "Layer": 1
        }

    elif isinstance(agent, Goal):
        portrayal = {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Color": "black",
            "Layer": 2
        }

    elif isinstance(agent, Evacuator):
        portrayal = {
            "Shape": "rect",
            "w": 1,
            "h": 1,
            "Filled": "true",
            "Color": "orange",
            "Layer": 2
        }

    return portrayal


# Init stuff

# width = 50
# height = 50

# N = int(0.25 * width * height)
# N = 10
# fire_radius = np.inf
# social_radius = width // 10
# p_spreading = 0.2
# p_spreading_environment = 0.3
# p_env_knowledge_params = [3/25, 17/25] # uniform, threshold 1 (no knowledge), threshold 2 (one door known)
# evacuator_radius = social_radius * 4

# exits = [ {"location": (0, height - 1), "radius": width // 2},
#           {"location": (width - 1, 0), "radius": width // 2},
#           {"location": (width - 1, height - 1), "radius": width // 2}]
# grid = CanvasGrid(portrayal, width, height)

# server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": width, "height": height, "N": N, 'p_env_knowledge_params': p_env_knowledge_params, "fire_radius": fire_radius, 'social_radius': social_radius, 'p_spreading': p_spreading, 'p_spreading_environment': p_spreading_environment, 'exits': exits, 'evacuator_present':False, 'evacuator_radius':evacuator_radius})
# server.port = 9986
# server.launch()

# data = server.model.datacollector.get_model_vars_dataframe()
# data.to_csv("agents_removed_per_step.csv", index=False)

# print("Data saved successfully!")
