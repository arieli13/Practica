import os
import neat
import math
import neat.visualize as visualize
import sys

sys.path.append("../../classes/Dataset")
sys.path.append("../../classes/Log")
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
from ListClassicDataset import ListClassicDataset
from LogString import LogString
from OwnReporter import OwnReporter

dataset_path_train = "../../datasets_angle_distance/cart_leftArmMovement_train.csv"
dataset_path_test = "../../datasets_angle_distance/cart_leftArmMovement.csv"

train_registers = 500
generations = 10

dataset_train = ListClassicDataset(dataset_path_train, 1, 4, 2, ",", 1, train_registers)
dataset_test = ListClassicDataset(dataset_path_test, 1, 4, 2, ",", 1)

def test(winner_net):
    error = 0.0
    distance_difference = 0.0
    angle_difference = 0.0
    log_predictions = LogString("./predictions.csv", "w+", "number,predicion_x,predicion_y,label_x,label_y\n", ",")
    while not dataset_test.dataset_out_of_range():
        xi, xo = dataset_test.get_next()
        xi, xo = xi[0], xo[0]
        output = winner_net.activate(xi)
        distance_difference += (output[0]-xo[0])**2
        angle_difference += (output[1]-xo[1])**2
        log_predictions.log_string([output[0], output[1], xo[0], xo[1]])
    error = distance_difference + angle_difference
    error = error/(2*dataset_test.get_size())
    error = math.sqrt(error)
    print("RMSE: %f"%(error))
    log_predictions.close_file()

def fitness_function(genomes, config):

    for genome_id, genome in genomes:
        distance_difference = 0
        angle_difference = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        dataset_train.restore_index()
        while not dataset_train.dataset_out_of_range():
            xi, xo = dataset_train.get_next()
            xi, xo = xi[0], xo[0]
            output = net.activate(xi)
            distance_difference += (output[0]-xo[0])**2
            angle_difference += (output[1]-xo[1])**2
        error = distance_difference + angle_difference
        error /= (train_registers*2)
        error = math.sqrt(error)

        genome.fitness = error*-1 #max


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(OwnReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="./checkpoints/ckpt-"))

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

    # Run for up to 300 generations.
    winner = p.run(fitness_function, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {-1:'x_t', -2: 'y_t', -3: 'x_act', -4:'y_act', 0:'x_t+1', 1: 'y_t+1'}
    visualize.draw_net(config, winner, True, node_names=node_names, filename="./reports/Digraph.gv")
    test(winner_net)
    visualize.plot_stats(stats, ylog=False, view=True, filename="./reports/avg_fitness.svg")
    visualize.plot_species(stats, view=True, filename="./reports/species.svg")


def main():
    global dataset_train, dataset_test
    local_dir = "./"
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
    

if __name__ == '__main__':
    main()