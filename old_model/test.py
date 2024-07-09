from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np

class CrowdAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.goals = [
            {"location": (0, model.grid.height - 1), "priority": 1},
            {"location": (model.grid.width - 1, 0), "priority": 2},
            {"location": (model.grid.width - 1, model.grid.height - 1), "priority": 2}
        ]
        self.current_goal = None

        self.knowledge_of_disaster = True
        self.knowledge_of_environment = True
        self.at_goal_timer = 1

        # Social force parameters
        self.desired_speed = 1.0
        self.relaxation_time = 0.5
        self.agent_radius = 0.5
        self.A = 2000  # Strength of repulsive force
        self.B = 0.08  # Range of repulsive force

    def step(self):
        """
        Performs a step in the agent's behavior.
        """
        if not self.knowledge_of_disaster:
            self.stand_still()
        else:
            if self.knowledge_of_environment:
                self.current_goal = self.goals[0]  # Set a default goal
                self.move_towards_goal()
            else:
                self.random_movement()

    def move_towards_goal(self):
        """
        Moves the agent towards its current goal.
        """
        if self.current_goal is None:
            return  # No goal to move towards

        # Calculate forces
        force_goal = self.calculate_goal_force()
        force_agents = self.calculate_agent_force()
        force_obstacles = self.calculate_obstacle_force()

        # Combine forces
        total_force = force_goal + force_agents + force_obstacles

        # Calculate the best move in the neighborhood based on combined forces
        self.make_best_move(total_force)

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
            print(f"Number of steps: {self.model.schedule.steps}")

    def calculate_goal_force(self):
        """
        Calculate the force towards the goal.
        """
        goal_x, goal_y = self.current_goal["location"]
        desired_direction = np.array([goal_x - self.pos[0], goal_y - self.pos[1]])
        norm = np.linalg.norm(desired_direction)
        if norm > 0:
            desired_direction /= norm
        desired_velocity = self.desired_speed * desired_direction
        current_velocity = np.array([0, 0])  # Assuming agents start from rest
        force_goal = (desired_velocity - current_velocity) / self.relaxation_time
        return force_goal

    def calculate_agent_force(self):
        """
        Calculate the repulsive force from other agents.
        """
        force = np.array([0.0, 0.0])
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                distance = np.linalg.norm(np.array(self.pos) - np.array(agent.pos))
                direction = np.array(self.pos) - np.array(agent.pos)
                if distance > 0:
                    force += self.A * np.exp((self.agent_radius - distance) / self.B) * (direction / distance)
        return force

    def calculate_obstacle_force(self):
        """
        Calculate the repulsive force from obstacles (fires).
        """
        force = np.array([0.0, 0.0])
        for fire in self.model.fire:
            distance = np.linalg.norm(np.array(self.pos) - np.array(fire.pos))
            direction = np.array(self.pos) - np.array(fire.pos)
            if distance > 0:
                force += self.A * np.exp((self.agent_radius - distance) / self.B) * (direction / distance)
        return force

    def make_best_move(self, total_force):
        """
        Make the best move in the neighborhood based on the combined forces.
        """
        # Get neighboring cells
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height]

        # Calculate the potential for each neighboring cell
        best_move = self.pos
        best_potential = float('inf')

        for neighbor in valid_neighbors:
            if self.model.grid.is_cell_empty(neighbor):
                # Calculate the potential considering the combined force
                potential = np.linalg.norm(np.array(neighbor) - (np.array(self.pos) + total_force))
                if potential < best_potential:
                    best_potential = potential
                    best_move = neighbor

        # Move to the best neighboring cell
        self.model.grid.move_agent(self, best_move)

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

    def __init__(self, width, height, N, goal_radius, fire_radius, fire_locations):
        """
        Initializes a CrowdModel object.

        Args:
            width (int): The width of the model's grid.
            height (int): The height of the model's grid.
            N (int): The number of agents in the model.
            goal_radius (float): The radius of the goal area.
        """

        assert len(fire_locations) < ((width * height) - N) / 2, 'Too many fire locations for amount of agents'

        self.num_agents = N
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.goal_radius = goal_radius
        self.fire_radius = fire_radius
        
        self.running = True  # Initialize the running state

        self.datacollector = DataCollector(
            {"Agents Removed": lambda m: m.num_agents_removed}
        )
        self.num_agents_removed = 0  # Track the number of agents removed

        # Create a fire
        # self.fire = []
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
        self.num_agents_removed = self.num_agents - len(self.schedule.agents)
        self.datacollector.collect(self)
        self.schedule.step()

class Hazard(Agent):
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

server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 20, "height": 20, "N": 100, "goal_radius": 10, "fire_radius": 10, "fire_locations": [[0,0], [0,1], [0,2]]})
server.port = 9998
server.launch()

data = server.model.datacollector.get_model_vars_dataframe()
data.to_csv("agents_removed_per_step.csv", index=False)

print("Data saved successfully!")
