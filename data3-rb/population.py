import numpy as np
from copy import deepcopy
from individual import Individual


class Population:
    def __init__(self, population_size, filename, nr_of_rules):
        self.individuals_list = []
        self.offsprings = []

        data = []
        with open(filename) as f:
            for line in f:
                l = line.split(' ')
                condition = np.array(l[:-1], dtype=np.float32)
                action = l[-1]
                action = int(action.replace("\n", ""))
                data.append([condition, action])
        self.data = data[:1000]
        self.validate_data = data[1000:]

        individual_size = len(self.data[0][0])
        for i in range(population_size):
            new_individual = Individual(individual_size, nr_of_rules)
            self.individuals_list.append(new_individual)

    def evaluate(self):
        for individual in self.individuals_list:
            individual.evaluate(self.data)

    def get_max_fitness(self):
        return max(c.fitness for c in self.individuals_list)

    def get_average_fitness(self):
        return sum(c.fitness for c in self.individuals_list)/len(self.individuals_list)

    def get_target_fitness(self):
        return len(self.data)

    def selection(self):
        offsprings = []
        population_size = len(self.individuals_list)

        for i in range(population_size):
            indexes = np.random.rand(2) * (population_size - 1)

            index1 = int(indexes[0])
            index2 = int(indexes[1])

            if self.individuals_list[index1].fitness > self.individuals_list[index2].fitness:
                offspring = deepcopy(self.individuals_list[index1])
            else:
                offspring = deepcopy(self.individuals_list[index2])

            offsprings.append(offspring)

        self.offsprings = offsprings

    def crossover(self, chance):
        for i in range(0, len(self.offsprings), 2):
            if np.random.rand() < chance:
                gene1 = deepcopy(self.offsprings[i].rules)
                gene2 = deepcopy(self.offsprings[i+1].rules)

                index = np.random.randint(1, len(gene1))

                aux = gene1[index:]
                gene1 = gene1[:index] + gene2[index:]
                gene2 = gene2[:index] + aux

                self.offsprings[i].rules = gene1
                self.offsprings[i+1].rules = gene2

    def mutation(self, chance):
        for offspring in self.offsprings:
            offspring.mutate(chance)

    def replace_population(self):
        # extract worst individual and replace with best
        index_min = np.argmin([c.fitness for c in self.offsprings])
        index_max = np.argmax([c.fitness for c in self.individuals_list])
        self.offsprings[index_min] = deepcopy(self.individuals_list[index_max])

        # copy the rest
        self.individuals_list = self.offsprings

    def get_max_fitness_individual(self):
        index = np.argmax([c.fitness for c in self.individuals_list])

        return self.individuals_list[index]

    def validate(self):
        index_max = np.argmax([c.fitness for c in self.individuals_list])
        best = self.individuals_list[index_max]
        best_fitness = best.fitness
        best.evaluate(self.validate_data)
        validate_fitness = best.fitness
        best.fitness = best_fitness

        return validate_fitness
