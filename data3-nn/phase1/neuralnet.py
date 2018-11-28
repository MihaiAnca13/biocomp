from phase1.other_functions import *
from keras.callbacks import TensorBoard

FILENAME = "../data3.txt"
LAYER_SIZE = 62
TRAINING_PERCENTAGE = 0.5
EPOCHS = 200
BATCH_SIZE = 32
NAME = "Assignment4"


data = read_file(FILENAME)
model = create_model(input_dim=len(data[0])-1, layer_size=LAYER_SIZE)

model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

print("Total data available: "+str(len(data)))
print(str(int(TRAINING_PERCENTAGE*len(data)))+" used for training")

X_train, Y_train = prepare_data(data[:int(TRAINING_PERCENTAGE * len(data))])
X_test, Y_test = prepare_data(data[int(TRAINING_PERCENTAGE * len(data)):])

model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard])

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

print(model.metrics_names, score)
