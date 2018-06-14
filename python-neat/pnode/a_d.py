import os
import neat
import math
import neat.visualize as visualize
import sys

sys.path.append("../../classes/")
sys.path.append("../../classes/Log")
from LogString import LogString
from OwnReporter import OwnReporter
from OwnCheckpointer import OwnCheckpointer
import Graphics as gf
import time

dataset_path = "../../pnodes/datasets/normalized/pnode0.csv"

train_registers = 500
generations = 10
memory_size = 100
last_checkpoint = ""
memory = []

time_log = LogString("./times_log.csv", "w+", "iteration,seconds\n", ",")
best_fit_log = LogString("./best_fit_log.csv", "w+", "iteration,value\n", ",")

def read_datasets(path, training_registers, skip):
    train = []
    validation = []
    test = []
    with open(path) as f:
        lines = f.readlines()[skip:]
        lines =  [ [[y[:8]], [y[8:]]] for y in [[float(k) for k in i.split(",")] for i in lines ]]
        train = lines[:training_registers]
        validation = lines[training_registers:]
        test = lines[:]
    return train, validation, test

dataset_train, _, dataset_test = read_datasets(dataset_path, train_registers, 1)

def test(winner_net):
    error = 0.0
    error_sq = 0.0
    log_predictions = LogString("./predictions.csv", "w+", "number,predicion_x,predicion_y,label_x,label_y\n", ",")
    for xi, xo in dataset_test:
        xi, xo = xi[0], xo[0]
        output = winner_net.activate(xi)
        mse = (output[0]-xo[0])**2
        error += mse
        error_sq += mse**2
        log_predictions.log_string([output[0], xo[0]])

    error /= len(dataset_test)
    error_sq /= len(dataset_test)
    var = error_sq + error**2
    stddev = math.sqrt(var)
    log_predictions.close_file()
    print("Cost: %f\tStddev: %f" % (error, stddev))


def fitness_function(genomes, config):
    global memory
    for genome_id, genome in genomes:
        error = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi,xo in memory:
            xi, xo = xi[0], xo[0]
            output = net.activate(xi)
            error += (output[0]-xo[0])**2
        error /= len(dataset_train)
        error = math.sqrt(error)
        genome.fitness = error*-1 #max

winner_net = None
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join("./", "config.txt"))
p = neat.Population(config)
p.add_reporter(OwnReporter())
def run(iteration):

    start_time = time.time()
    print("Comienzo")
    winner = p.run(fitness_function, generations)
    finish_time = time.time()
    print("Termino")
    total_time = finish_time-start_time
    # Display the winning genome.
    print('\nIteration: %d, Best genome: %f'%(iteration, winner.fitness))
    if iteration >= 499:
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    best_fit_log.log_string([abs(winner.fitness)])
    #own_checkpointer.save_checkpoint(p.config, p.population, p.species)
    return total_time

def stochastic_train_memory():
    global memory, dataset_train, winner_net
    #own_checkpointer = OwnCheckpointer("./checkpoints/ckpt-")
    iteration = 0
    for i in dataset_train:
        memory.append(i)
        if len(memory) > memory_size:
            memory = memory[1:]
        total_time = run(iteration)
        time_log.log_string([total_time])
        iteration += 1
    test(winner_net)
    time_log.close_file()
    best_fit_log.close_file()
    gf.plot_csv("./predictions.csv", ",", 0, [1, 2], "Iteration", "Value", "Predictions log", ["r+", "ko"], True, None)
    gf.plot_csv("./times_log.csv", ",", 0, [1], "Iteration", "Seconds", "Times log", ["ko"], True, "./times_log_img.png")


def main():
    global dataset_train, dataset_test
    stochastic_train_memory()

if __name__ == '__main__':
    main()
