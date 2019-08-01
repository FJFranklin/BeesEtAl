import numpy as np

class Base_Automaton(object):

    def __init__(self, count, reward=0.10, punish=0.05):
        self.count    = count
        self.__reward = reward
        self.__punish = punish
        self.cells    = np.ones(self.count) / self.count

    def cell(self):
        return np.random.choice(self.count, p=self.cells)

    def cool(self, cooling_factor):
        self.__reward = self.__reward * cooling_factor
        self.__punish = self.__punish * cooling_factor

    def reward(self, cell):
        for p in range(0, self.count):
            if p == cell:
                self.cells[p] = self.cells[p] + self.__reward * (1 - self.cells[p])
            else:
                self.cells[p] = self.cells[p] * (1 - self.__reward)

        self.cells[0] = 1 - sum(self.cells[1:])

    def punish(self, cell):
        for p in range(0, self.count):
            if p == cell:
                self.cells[p] = self.cells[p] * (1 - self.__punish)
            else:
                self.cells[p] = self.cells[p] * (1 - self.__punish) + self.__punish / (self.count - 1)

        self.cells[0] = 1 - sum(self.cells[1:])
