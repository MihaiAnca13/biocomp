import numpy as np


class Individual:
    def __init__(self, size, nr_of_rules):
        self.rules = []
        self.fitness = 0
        self.size = size
        self.nr_of_rules = nr_of_rules

        for i in range(nr_of_rules):
            condition_set = []
            for j in range(size):
                # generate upper and lower limit
                condition = np.random.rand(2).astype(np.float32)
                condition_set.append(condition)

            action = np.random.randint(0, 2, dtype=np.int8)

            self.rules.append([condition_set, action])

    def evaluate(self, data):
        self.fitness = 0

        for condition, action in data:
            for self_condition, self_action in self.rules:
                equal = True
                for i in range(len(self_condition)):
                    if self_condition[i][0] > self_condition[i][1]:
                        self_condition[i][0] += self_condition[i][1]
                        self_condition[i][1] = self_condition[i][0] - self_condition[i][1]
                        self_condition[i][0] -= self_condition[i][1]
                    if condition[i] < self_condition[i][0] or condition[i] > self_condition[i][1]:
                        equal = False
                        break
                if equal:
                    if self_action == action:
                        self.fitness += 1
                    break

    def mutate(self, chance):
        for i in range(self.nr_of_rules):
            for j in range(len(self.rules[i][0])):
                for l in range(len(self.rules[i][0][j])):
                    if np.random.rand() < chance:
                        self.rules[i][0][j][l] += np.random.uniform(-0.1, 0.1)

            if np.random.rand() < chance:
                if self.rules[i][1] == 0:
                    self.rules[i][1] = 1
                else:
                    self.rules[i][1] = 0
