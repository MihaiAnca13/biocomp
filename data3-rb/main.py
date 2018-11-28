from population import *
import matplotlib.pyplot as plt
from time import time

plt.style.use('ggplot')

POPULATION_SIZE = 50
FILENAME = '../data3.txt'
RULES = 20
PC = 1
PM = 0.0035


def main(population_size, filename, rules, crossover_chance, mutation_chance):
    x_vec = []
    max_vec = []
    mean_vec = []
    validate_vec = []

    population = Population(population_size, filename, rules)
    population.evaluate()

    iteration = 0
    while population.get_max_fitness() < population.get_target_fitness():
        iteration += 1
        population.selection()
        population.crossover(crossover_chance)
        population.mutation(mutation_chance)
        population.replace_population()
        population.evaluate()
        print(f"Iteration: {iteration} | Average: {population.get_average_fitness()/population.get_target_fitness()*100}% | Max: {population.get_max_fitness()/population.get_target_fitness()*100}%")

        # update plot
        plt.clf()
        x_vec.append(iteration)
        max_vec.append(population.get_max_fitness())
        mean_vec.append(population.get_average_fitness())
        validate_vec.append(population.validate())
        plt.figure(1)
        line, = plt.plot(x_vec, max_vec, 'b')
        line2, = plt.plot(x_vec, mean_vec, 'r')
        plt.figure(2)
        line3, = plt.plot(x_vec, validate_vec, 'g')
        plt.show(block=False)
        plt.pause(0.0001)

    print(population.get_max_fitness_individual().rules)


if __name__ == '__main__':
    main(POPULATION_SIZE, FILENAME, RULES, PC, PM)
