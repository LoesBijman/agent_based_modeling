from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np

class CrowdAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.goal = (0, model.grid.height - 1)  # Exit at top-left corner

    def step(self):
        self.move_towards_goal()

    def move_towards_goal(self):
        x, y = self.pos
        goal_x, goal_y = self.goal

        dx = np.sign(goal_x - x)
        dy = np.sign(goal_y - y)

        new_position = (x + dx, y + dy)

        if self.model.grid.is_cell_empty(new_position):
            self.model.grid.move_agent(self, new_position)

class CrowdModel(Model):
    def __init__(self, width, height, N):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

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

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)

server = ModularServer(CrowdModel, [grid], "Crowd Model", {"width": 10, "height": 10, "N": 10})
server.port = 8521
server.launch()
