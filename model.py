from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
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
		check_goal_knowledge(): Checks if the agent has knowledge of its current goal.
		move_towards_goal(): Moves the agent towards its current goal.
	"""

	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)

		# Define the goals that the agent can move towards
		self.goals = [
			{"location": (0, model.grid.height - 1), "priority": 1},
			{"location": (model.grid.width - 1, 0), "priority": 2},
			{"location": (model.grid.width - 1, model.grid.height - 1), "priority": 2}
		]
		self.current_goal = None

	def step(self):
		"""
		Performs a step in the agent's behavior.
		"""
		self.check_goal_knowledge()
		self.move_towards_goal()

	def check_goal_knowledge(self):
		"""
		Checks if the agent has knowledge of its current goal.
		"""
		known_goals = [self.goals[0]]
		for goal in self.goals:

			# Eucledian distance to goal
			distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(goal["location"]))

			# Check if the agent is within the goal radius
			if distance_to_goal < self.model.goal_radius:
				known_goals.append(goal)

		# Sort known goals by priority and select the highest priority goal
		self.current_goal = sorted(known_goals, key=lambda x: x["priority"], reverse=True)[0]

	def move_towards_goal(self):
		"""
		Moves the agent towards its current goal.
		"""
		# own position and goal position
		x, y = self.pos
		goal_x, goal_y = self.current_goal["location"]

		# change in position
		dx = np.sign(goal_x - x)
		dy = np.sign(goal_y - y)

		# new position
		new_position = (x + dx, y + dy)

		# move agent to new position if it is empty
		if self.model.grid.is_cell_empty(new_position):
			self.model.grid.move_agent(self, new_position)
		else:
			# check for second best option if the first option is not available (not sure if this works as intended)
			if dx == 0:
				new_position = (x, y + dy)
			elif dy == 0:
				new_position = (x + dx, y)
			if self.model.grid.is_cell_empty(new_position):
				self.model.grid.move_agent(self, new_position)

		# Check if the agent has reached the goal, and remove it from the model if it has
		if self.pos == self.current_goal["location"]:
			print(f"Agent {self.unique_id} reached the goal!")
			self.model.grid.remove_agent(self)
			self.model.schedule.remove(self)

		# If all agents have reached the goal, stop the model
		if len(self.model.schedule.agents) == 0:
			self.model.running = False
			# print the number of steps it took for all agents to reach the goal
			print(f"Number of steps: {self.model.schedule.steps}")


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

	def __init__(self, width, height, N, goal_radius):
		"""
		Initializes a CrowdModel object.

		Args:
			width (int): The width of the model's grid.
			height (int): The height of the model's grid.
			N (int): The number of agents in the model.
			goal_radius (float): The radius of the goal area.
		"""
		self.num_agents = N
		self.grid = MultiGrid(width, height, True)
		self.schedule = RandomActivation(self)
		self.goal_radius = goal_radius

		# Create agents and place them in the model
		for i in range(self.num_agents):
			x = self.random.randint(0, width - 1)
			y = self.random.randint(0, height - 1)
			agent = CrowdAgent(i, self)
			self.schedule.add(agent)
			self.grid.place_agent(agent, (x, y))

	def step(self):
		"""
		Advances the model by one step.
		"""
		self.schedule.step()

def agent_portrayal(agent):
	"""
	Returns the visualization for a given agent.

	Parameters:
	agent (object): The agent for which the portrayal is being generated.

	Returns:
	dict: The portrayal dictionary containing the agent's shape, color, size, and layer.
	"""
	portrayal = {
		"Shape": "circle",
		"Filled": "true",
		"r": 0.5,
		"Color": "blue",
		"Layer": 0  
	}
	return portrayal

# Init stuff
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 20, "height": 20, "N": 100, "goal_radius": 10})
server.port = 9998
server.launch()
