from utilities import *


class Expert:
    def __init__(self):
        self.item = None
        self.goal = None

    def update_item_and_goal(self, item: Item, goal: Item):
        self.item = item
        self.goal = goal

    def calculate_poke(self):
        pass

