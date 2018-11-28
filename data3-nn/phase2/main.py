import matplotlib.pyplot as plt
import tkinter
import csv
from help_functions import extract_best, selection, crossover, mutation
from phase2.new_functions import *
from phase1.other_functions import read_file
from time import time

POPULATION_SIZE = 20
CANDIDATE_LENGHT = 20
MAX_ITERATIONS = 100
PC = 0.5
PM = 0.1
LEN_LIMIT = 100
ROUNDS = 1
DEBUG = True
FILENAME = f"P{POPULATION_SIZE}-L{CANDIDATE_LENGHT}-C{int(PC*100)}-M{int(PM*100)}-{round(time()%10000)}"
WRITE_FILE = False
DATA_FILENAME = '../data3.txt'

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

x_vec = []
max_vec = []
mean_vec = []

fitness_array = []

if __name__ == '__main__':

    print('Starting')

    data = read_file(DATA_FILENAME)

    if WRITE_FILE:
        with open(FILENAME + '.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Round', 'Best-Fitness', 'Average-Fitness'])

    for r in range(ROUNDS):
        population = pop_initialise(POPULATION_SIZE, CANDIDATE_LENGHT)

        iteration = 0
        max_fitness = max(c.fitness for c in population)

        offsprings = np.copy(population)

        while iteration < MAX_ITERATIONS:
            # update iteration number
            iteration += 1

            # extract best individual
            best_individual = extract_best(population)

            # initialise new population with the new offsprings
            population = np.array(offsprings)

            # extract worst individual and replace with best
            index_min = np.argmin(c.fitness for c in population)
            population[index_min] = best_individual

            # calculate fitness
            for c in population:
                evaluate(c, data)

            # update stats
            max_fitness = max(c.fitness for c in population)
            total_fitness = sum(c.fitness for c in population)
            mean_fitness = np.mean([c.fitness for c in population])

            # update plot
            plt.clf()

            if len(x_vec) > LEN_LIMIT:
                x_vec.pop(0)
                mean_vec.pop(0)
                max_vec.pop(0)

            x_vec.append(iteration)
            max_vec.append(max_fitness)
            mean_vec.append(mean_fitness)
            line, = plt.plot(x_vec, max_vec, 'b')
            line2, = plt.plot(x_vec, mean_vec, 'r')

            if WRITE_FILE:
                with open(FILENAME + '.csv', 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([r, max_fitness, mean_fitness])

            if DEBUG:
                plt.show(block=False)
                plt.pause(0.01)
                print(f"Iteration {iteration} - Max: {max_fitness} - Total: {total_fitness}")

            # selection
            offsprings = selection(population)

            # crossover
            for i in range(0, POPULATION_SIZE, 2):
                # select two individuals
                parent1 = offsprings[i]
                parent2 = offsprings[i+1]

                # apply crossover
                offsprings[i], offsprings[i+1] = crossover(parent1.value, parent2.value, chance=PC)

            # mutation
            for i in range(POPULATION_SIZE):
                value = np.copy(offsprings[i])

                # apply mutation
                new_candidate = mutation(value, chance=PM)

                # reconstruct object
                offsprings[i] = Candidate(CANDIDATE_LENGHT, value=new_candidate)

        # print best candidate
        for c in population:
            if c.fitness == max_fitness:
                print(f'Round: {r}\n{c.value}')
                x_vec = []
                max_vec = []
                mean_vec = []
                break

    if DEBUG:
        input()
    print('Done')


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# 992287