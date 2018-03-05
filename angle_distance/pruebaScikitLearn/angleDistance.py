from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

model = MLPRegressor(hidden_layer_sizes=(100, ), activation="relu", solver="adam", alpha=0.0001, batch_size="auto", learning_rate="adaptive", learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

train_X = []
train_Y = []
file = open("../dataset/data_100.txt", "r")
for line in file:
    line = line.split(" ")
    line = map(float, line)
    train_X.append(np.array(line[:3]))
    train_Y.append(np.array(line[3:]))

#2.005866135438498 -1.6027494869492036 -0.7772030397378362 1.9723006980591289 -1.6213815594376575
model = model.fit(train_X, train_Y)
