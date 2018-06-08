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

train_registers = 250
generations = 500
memory_size = 100
last_checkpoint = ""
memory = []

time_log = LogString("./times_log.csv", "w+", "iteration,seconds\n", ",")

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
    log_predictions = LogString("./predictions.csv", "w+", "number,predicion_x,predicion_y,label_x,label_y\n", ",")
    for xi, xo in dataset_test:
        xi, xo = xi[0], xo[0]
        output = winner_net.activate(xi)
        error += (output[0]-xo[0])**2
        log_predictions.log_string([output[0], xo[0]])
    error /= len(dataset_test)
    error = math.sqrt(error)
    print("RMSE: %f"%(error))
    log_predictions.close_file()

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


def run(config_file, own_checkpointer):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    #checkpointer_reporter = neat.Checkpointer(100, filename_prefix="./checkpoints/ckpt-")
    #p.add_reporter(own_checkpointer)
    last_checkpoint = own_checkpointer.restore_checkpoint()
    if last_checkpoint is not None:
        p = last_checkpoint

    p.add_reporter(OwnReporter())
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)

    winner = p.run(fitness_function, generations)

    # Display the winning genome.
    print('\nMemory_size: %d, Best genome: %f'%(len(memory), winner.fitness))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    own_checkpointer.save_checkpoint(p.config, p.population, p.species)
    node_names = {-1:'a', -2: 'x', -3: 'c', -4:'v', -5: 'b', -6: 'n', -7: 'q', 0: 'w', 1: 'n'}
    #visualize.draw_net(config, winner, True, node_names=node_names, filename="./reports/Digraph.gv")
    return winner_net
    
    #test(winner_net)
    #visualize.plot_stats(stats, ylog=False, view=True, filename="./reports/avg_fitness.svg")
    #visualize.plot_species(stats, view=True, filename="./reports/species.svg")

def stochastic_train_memory(config_file):
    global memory, dataset_train
    winner_net = None
    own_checkpointer = OwnCheckpointer("./checkpoints/ckpt-")
    for i in dataset_train:
        memory.append(i)
        if len(memory) > memory_size:
            memory = memory[1:]
        start_time = time.time()
        winner_net = run(config_file, own_checkpointer)
        finish_time = time.time()
        time_log.log_string([finish_time-start_time])
    test(winner_net)
    time_log.close_file()
    gf.plot_csv("./predictions.csv", ",", 0, [1, 2], "Iteration", "Value", "Predictions log", ["r+", "ko"], True, None)
    gf.plot_csv("./times_log.csv", ",", 0, [1], "Iteration", "Seconds", "Times log", ["ko"], True, "./times_log_img.png")


def main():
    global dataset_train, dataset_test
    local_dir = "./"
    config_path = os.path.join(local_dir, "config.txt")
    stochastic_train_memory(config_path)

if __name__ == '__main__':
    main()
