import numpy as np


class Individual:
    def __init__(self, size, nr_of_rules):
        self.rules = []
        self.fitness = 0
        self.size = size
        self.nr_of_rules = nr_of_rules

        for i in range(nr_of_rules):
            condition = np.random.randint(0, 3, size, dtype=np.int8)
            action = np.random.randint(0, 2, dtype=np.int8)

            self.rules.append([condition, action])

    def evaluate(self, data):
        self.fitness = 0

        for condition, action in data:
            for self_condition, self_action in self.rules:
                equal = True
                for i in range(len(self_condition)):
                    if self_condition[i] != 2 and self_condition[i] != int(condition[i]):
                        equal = False
                        break
                if equal:
                    if self_action == int(action):
                        self.fitness += 1
                    break

    def to_binary(self):
        b_array = []
        for condition, action in self.rules:
            for item in condition:
                b_array.append(item)
            b_array.append(action)

        return b_array

    def from_binary(self, b_array):
        self.rules = []
        for i in range(0, len(b_array), int(len(b_array)/self.nr_of_rules)):
            condition = np.array([b_array[j] for j in range(i, i+self.size)])
            action = b_array[i+self.size]

            self.rules.append([condition, action])

    def mutate(self, chance):
        for i in range(self.nr_of_rules):
            for j in range(len(self.rules[i][0])):
                if np.random.rand() < chance:
                    previous = self.rules[i][0][j]
                    while self.rules[i][0][j] == previous:
                        self.rules[i][0][j] = np.random.randint(3)
            if np.random.rand() < chance:
                if self.rules[i][1] == 0:
                    self.rules[i][1] = 1
                else:
                    self.rules[i][1] = 0
