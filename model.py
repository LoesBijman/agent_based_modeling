from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
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
		self.knows_goal = False
		self.current_goal = None

	def step(self):
		self.check_goal_knowledge()
		self.move_towards_goal()

	def check_goal_knowledge(self):
		known_goals = [self.goals[0]]
		for goal in self.goals:
			distance_to_goal = np.linalg.norm(np.array(self.pos) - np.array(goal["location"]))
			if distance_to_goal < self.model.goal_radius:
				known_goals.append(goal)
		
		if known_goals:
			# Sort known goals by priority and select the highest priority goal
			self.current_goal = sorted(known_goals, key=lambda x: x["priority"], reverse=True)[0]
			self.knows_goal = True
		else:
			self.knows_goal = False

	def move_towards_goal(self):
		if self.knows_goal:
			self.move_towards_known_goal()
		else:
			self.follow_crowd()

	def move_towards_known_goal(self):
		x, y = self.pos
		goal_x, goal_y = self.current_goal["location"]

		dx = np.sign(goal_x - x)
		dy = np.sign(goal_y - y)

		new_position = (x + dx, y + dy)

		if self.model.grid.is_cell_empty(new_position):
			self.model.grid.move_agent(self, new_position)
		else:
		# check for second best option if the first option is not available
			if dx == 0:
				new_position = (x, y + dy)
			elif dy == 0:
				new_position = (x + dx, y)
			if self.model.grid.is_cell_empty(new_position):
				self.model.grid.move_agent(self, new_position)

		if self.pos == self.current_goal["location"]:
			print(f"Agent {self.unique_id} reached the goal!")
			self.model.grid.remove_agent(self)
			self.model.schedule.remove(self)

		# If all agents have reached the goal, stop the model
		if len(self.model.schedule.agents) == 0:
			self.model.running = False
			# print the number of steps it took for all agents to reach the goal
			print(f"Number of steps: {self.model.schedule.steps}")

	def follow_crowd(self):
		# get crowd density around the agent
		x, y = self.pos
		neighbors = self.model.grid.get_neighbors((x, y), moore=True, include_center=False, radius=5)
		# move towards the average position of the neighbors
		if len(neighbors) > 0:
			avg_x = np.mean([pos[0] for pos in [neighbor.pos for neighbor in neighbors]])
			avg_y = np.mean([pos[1] for pos in [neighbor.pos for neighbor in neighbors]])
			new_position = (int(avg_x), int(avg_y))
			if self.model.grid.is_cell_empty(new_position):
				self.model.grid.move_agent(self, new_position)

class CrowdModel(Model):
	def __init__(self, width, height, N, goal_radius):
		self.num_agents = N
		self.grid = MultiGrid(width, height, True)
		self.schedule = RandomActivation(self)
		self.goal_radius = goal_radius

		for i in range(self.num_agents):
			x = self.random.randint(0, width - 1)
			y = self.random.randint(0, height - 1)
			agent = CrowdAgent(i, self)
			self.schedule.add(agent)
			self.grid.place_agent(agent, (x, y))

	def step(self):
		self.schedule.step()

def agent_portrayal(agent):
	portrayal = {
		"Shape": "circle",
		"Filled": "true",
		"r": 0.5,
		"Color": "blue",
		"Layer": 0  
	}
	return portrayal

grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 20, "height": 20, "N": 100, "goal_radius": 10})
server.port = 9998
server.launch()
