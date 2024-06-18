from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np


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

        # Define the goals that the agent can move towards
        # self.goals = [
        #     {"location": (0, model.grid.height - 1), "priority": 1},
        #     {"location": (model.grid.width - 1, 0), "priority": 2},
        #     {"location": (model.grid.width - 1, model.grid.height - 1), "priority": 2}
        # ]
        self.current_goal = None

        self.knowledge_of_disaster = False
        self.knowledge_of_environment = True
        self.at_goal_timer = 1

    def step(self):
        """
        Performs a step in the agent's behavior.
        """
        # Update the agent's knowledge of the disaster
        for fire in self.model.fire:
            if np.linalg.norm(np.array(self.pos) - np.array(fire.pos)) < self.model.fire_radius:
                # spread knowledge stochastically
                u = np.random.uniform(0,1)
                if u < self.model.p_spreading:
                    self.knowledge_of_disaster = True
        
        disaster_knowing_agents = [agent for agent in self.model.schedule.agents if isinstance(agent, CrowdAgent) and agent.knowledge_of_disaster]

        # Perform step
        if not self.knowledge_of_disaster:
            for dis_agent in disaster_knowing_agents:
                if np.linalg.norm(np.array(self.pos) - np.array(dis_agent.pos)) < self.model.social_radius:
                    self.knowledge_of_disaster = True
            self.stand_still()

        else:
            if self.knowledge_of_environment:
                self.current_goal = self.goals[0]  # Set a default goal
                self.move_towards_goal()
            else:
                # random exploration
                self.random_movement()

    def move_towards_goal(self):
        """
        Moves the agent towards its current goal.
        """
        if self.current_goal is None:
            return  # No goal to move towards

        # Own position and goal position
        x, y = self.pos
        goal_x, goal_y = self.current_goal["location"]

        # Calculate the distance to the goal for all neighboring cells
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height]
        distances = [(neighbor, np.linalg.norm(np.array(neighbor) - np.array(self.current_goal["location"])))
                     for neighbor in valid_neighbors if self.model.grid.is_cell_empty(neighbor)]
        
        # Move to the neighboring cell that is closest to the goal and empty
        if distances:
            next_move = min(distances, key=lambda t: t[1])[0]
            self.model.grid.move_agent(self, next_move)

        # Check if the agent has reached the goal, and remove it from the model if it has
        if self.pos == self.current_goal["location"]:
            if self.at_goal_timer == 0:
                print(f"Agent {self.unique_id} reached the goal!")
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
            elif self.at_goal_timer == 1:
                self.at_goal_timer -= 1


        # If all agents have reached the goal, stop the model
        if len(self.model.schedule.agents) == 0:
            self.model.running = False
            # print the number of steps it took for all agents to reach the goal
            print(f"Number of steps: {self.model.schedule.steps}")

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

    def __init__(self, width, height, N, goal_radius, fire_radius, fire_locations, social_radius, p_spreading):
        """
        Initializes a CrowdModel object.

        Args:
            width (int): The width of the model's grid.
            height (int): The height of the model's grid.
            N (int): The number of agents in the model.
            goal_radius (float): The radius of the goal area.
        """

        assert len(fire_locations) < ((width * height) - N) / 2, 'Too many fire locations for amount of agents'

        self.num_agents = N - len(fire_locations)
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.goal_radius = goal_radius
        self.fire_radius = fire_radius
        self.social_radius = social_radius
        self.p_spreading = p_spreading
        
        self.running = True  # Initialize the running state

        self.datacollector = DataCollector(
            {"Agents Removed": lambda m: m.num_agents_removed}
        )
        self.num_agents_removed = 0  # Track the number of agents removed

        # Create a fire
        for i, fire_loc in enumerate(fire_locations):
            x, y = fire_loc
            fire = Hazard(i, self)
            self.schedule.add(fire)
            self.grid.place_agent(fire, (x,y))

        # retrieve the fire locations
        self.fire = [agent for agent in self.schedule.agents if isinstance(agent, Hazard)]

        # Create agents and place them in the model
        for i in range(self.num_agents):
            x = self.random.randint(0, width - 1)
            y = self.random.randint(0, height - 1)
            while (x,y) in fire_locations:
                x = self.random.randint(0, width - 1)
                y = self.random.randint(0, height - 1)

            agent = CrowdAgent(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        """
        Advances the model by one step.
        """
        # Collect data on the number of CrowdAgents removed
        self.num_agents_removed = self.num_agents - sum(1 for agent in self.schedule.agents if isinstance(agent, CrowdAgent))
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Check if all CrowdAgents have reached their goals
        all_agents_reached_goal = all(agent.current_goal is None for agent in self.schedule.agents if isinstance(agent, CrowdAgent))
        if all_agents_reached_goal:
            self.running = False
            print(f"Number of steps: {self.schedule.steps}")

class Hazard(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Goal(Agent):
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
    return portrayal


# Init stuff
grid = CanvasGrid(portrayal, 20, 20, 500, 500)

server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 20, "height": 20, "N": 100, "goal_radius": 10, "fire_radius": 10, "fire_locations": [[0,0], [0,1], [0,2]], 'social_radius': 3, 'p_spreading': 0.5})
server.port = 9998
server.launch()

data = server.model.datacollector.get_model_vars_dataframe()
data.to_csv("agents_removed_per_step.csv", index=False)

print("Data saved successfully!")
