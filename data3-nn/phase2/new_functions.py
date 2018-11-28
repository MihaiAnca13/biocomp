from phase1.other_functions import *
from keras.callbacks import TensorBoard
from candidate import Candidate

FILENAME = "../data3.txt"


def binary_to_int(value):
    to_send = ""
    for i in range(len(value)):
        to_send += str(value[i])
    to_send = int(to_send, 2)
    return to_send


def evaluate(individual, data):
    # individual: layer_size + training_percentage + batch_size + epochs
    #               8-63        0.6-0.9             4-63            1-15
    #               6 bits      4 bits              6 bits          4 bits
    # total of 20 bits
    layer_size = binary_to_int(individual.value[:6])
    if layer_size < 8:
        layer_size = 8
    training_percentage = binary_to_int(individual.value[6:10])
    training_percentage = (training_percentage/15)*0.3+0.6
    batch_size = binary_to_int(individual.value[10:16])
    if batch_size < 4:
        batch_size = 4
    epochs = binary_to_int(individual.value[16:])
    if epochs < 1:
        epochs = 1

    model = create_model(input_dim=len(data[0]) - 1, layer_size=layer_size)

    # model.summary()
    name = f"GA-NN-{binary_to_int(individual.value)}"
    tensorboard = TensorBoard(log_dir=f"logs/{name}")

    X_train, Y_train = prepare_data(data[:int(training_percentage * len(data))])
    X_test, Y_test = prepare_data(data[int(training_percentage * len(data)):])

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[tensorboard])

    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)

    individual.fitness = int(score[1]*1000)


def pop_initialise(population_size, candidate_length):
    population = []

    for i in range(population_size):
        value = np.random.randint(2, size=[candidate_length])
        population.append(Candidate(candidate_length, value=value))

    return np.array(population)