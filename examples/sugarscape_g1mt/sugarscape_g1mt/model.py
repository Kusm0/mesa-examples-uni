from pathlib import Path
import mesa
import numpy as np
from typing import List, Optional, Union

from .resource_agents import Resource
from .trader_agents import Trader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


# Helper Functions
def flatten(list_of_lists: List[List]) -> List:
    """
    Collapses a list of lists into a single list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def geometric_mean(list_of_prices: List[float]) -> float:
    """
    Calculates the geometric mean of a list of prices.
    Returns 1.0 if list is empty to avoid errors in calculations.
    """
    if not list_of_prices:
        return 1.0
    return np.exp(np.log(list_of_prices).mean())


def get_trade(agent: mesa.Agent) -> Optional[List]:
    """
    Returns a list of trade partners if the agent is a Trader, otherwise None.
    """
    return agent.trade_partners if isinstance(agent, Trader) else None


class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with Traders.
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        initial_population: int = 200,
        endowment_min: int = 25,
        endowment_max: int = 50,
        metabolism_min: int = 1,
        metabolism_max: int = 5,
        vision_min: int = 1,
        vision_max: int = 5,
        enable_trade: bool = True,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.initial_population = initial_population
        self.endowment_min = endowment_min
        self.endowment_max = endowment_max
        self.metabolism_min = metabolism_min
        self.metabolism_max = metabolism_max
        self.vision_min = vision_min
        self.vision_max = vision_max
        self.enable_trade = enable_trade
        self.steps = 0
        self.running = True

        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Trader": lambda m: len(m.agents_by_type[Trader]),
                "Trade Volume": lambda m: self.get_total_trade_volume(m),
                "Price": lambda m: geometric_mean(
                    flatten([a.prices for a in m.agents_by_type[Trader]])
                ),
            },
            agent_reporters={"Trade Network": get_trade},
        )

        self.setup_landscape()
        self.initialize_population()

    def setup_landscape(self):
        """
        Initializes the resource landscape using a sugar distribution map.
        """
        sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        spice_distribution = np.flip(sugar_distribution, 1)

        for id, (_, (x, y)) in enumerate(self.grid.coord_iter()):
            max_sugar = sugar_distribution[x, y]
            max_spice = spice_distribution[x, y]
            resource = Resource(id, self, max_sugar, max_spice)
            self.grid.place_agent(resource, (x, y))

    def initialize_population(self):
        """
        Initializes the trader agents and places them on the grid.
        """
        for _ in range(self.initial_population):
            x, y = self.random.randrange(self.width), self.random.randrange(self.height)
            sugar, spice = self.random_uniform(self.endowment_min, self.endowment_max), self.random_uniform(self.endowment_min, self.endowment_max)
            metabolism_sugar = self.random_uniform(self.metabolism_min, self.metabolism_max)
            metabolism_spice = self.random_uniform(self.metabolism_min, self.metabolism_max)
            vision = self.random_uniform(self.vision_min, self.vision_max)

            trader = Trader(
                _,
                self,
                moore=False,
                sugar=sugar,
                spice=spice,
                metabolism_sugar=metabolism_sugar,
                metabolism_spice=metabolism_spice,
                vision=vision,
            )
            self.grid.place_agent(trader, (x, y))

    def random_uniform(self, min_val: int, max_val: int) -> int:
        """
        Returns a random integer between min_val and max_val.
        """
        return int(self.random.uniform(min_val, max_val + 1))

    def get_total_trade_volume(self, model) -> int:
        """
        Calculates the total volume of trades made by all traders.
        """
        return sum(len(a.trade_partners) for a in model.agents_by_type[Trader])

    def step(self):
        """
        Executes one step of the simulation, performing actions for resource and trader agents.
        """
        self.agents_by_type[Resource].do("step")
        self.process_traders()
        self.datacollector.collect(self)

        # Print or log agent trade records at the end of each step
        self.clean_up_trade_records()

    def process_traders(self):
        """
        Processes the movement, eating, and trade actions of trader agents.
        """
        trader_shuffle = self.agents_by_type[Trader].shuffle()

        for agent in trader_shuffle:
            agent.prices = []
            agent.trade_partners = []
            agent.move()
            agent.eat()
            agent.maybe_die()

        if self.enable_trade:
            for agent in trader_shuffle:
                agent.trade_with_neighbors()

    def clean_up_trade_records(self):
        """
        Cleans up the trade records to save memory.
        """
        agent_trades = self.datacollector._agent_records[self.steps]
        agent_trades = [agent for agent in agent_trades if agent[2] is not None]
        self.datacollector._agent_records[self.steps] = agent_trades

    def run_model(self, step_count: int = 1000):
        """
        Runs the model for the given number of steps.
        """
        for i in range(step_count):
            self.step()
            self.steps += 1
