from population import *
import matplotlib.pyplot as plt

plt.style.use('ggplot')

POPULATION_SIZE = 100
FILENAME = '../data2.txt'
RULES = 5
PC = 1
PM = 0.025
DEBUG = True
WRITE_FILE = False
MAX_ITERATION_RUN = 100


# done with 5 rules
# 0, 0, 2, 2, 2, 1, 2  1
# 1, 0, 2, 1, 2, 2, 2  1
# 1, 1, 1, 2, 2, 2, 2  1
# 0, 1, 2, 2, 1, 2, 2  1
# 2, 2, 2, 2, 2, 2, 2  0


def main(population_size, filename, rules, crossover_chance, mutation_chance, run_nr=None):
    x_vec = []
    max_vec = []
    mean_vec = []

    population = Population(population_size, filename, rules)
    population.evaluate()

    iteration = 0
    while population.get_max_fitness() < population.get_target_fitness():
        iteration += 1

        if run_nr is not None and iteration == MAX_ITERATION_RUN:
            break

        population.selection()
        population.crossover(crossover_chance)
        population.mutation(mutation_chance)
        population.replace_population()
        population.evaluate()

        if DEBUG:
            print(f"Iteration: {iteration} | Average: {population.get_average_fitness()} | Max: {population.get_max_fitness()}")

            if WRITE_FILE and run_nr:
                with open('average3.csv', 'a') as f:
                    f.write(f"{run_nr},{iteration},{population.get_max_fitness()},{population.get_average_fitness()}\n")

            # update plot
            plt.clf()
            x_vec.append(iteration)
            max_vec.append(population.get_max_fitness())
            mean_vec.append(population.get_average_fitness())
            line, = plt.plot(x_vec, max_vec, 'b')
            line2, = plt.plot(x_vec, mean_vec, 'r')
            plt.show(block=False)
            plt.pause(0.0001)

        if iteration%10 == 0:
            print(iteration)

    print("Rules used: " + str(rules))
    for r in population.get_max_fitness_individual().rules:
        print(r[0], r[1], "\n")

    return mean_vec


if __name__ == '__main__':
    main(POPULATION_SIZE, FILENAME, RULES, PC, PM)