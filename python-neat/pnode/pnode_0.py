import os
import neat
import math

#import visualize
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

dataset_path = "../pnodes/datasets/normalizado/pnode01_03000"
dataset_train_path = dataset_path+"_train.txt"
dataset_test_path = dataset_path+"_test.txt"
dataset_full_path = dataset_path+".txt"

dataset_train = []
dataset_test = []


def load_dataset(path):
    """
    Create a dataset from a csv.

    The path is of an existing csv, each line separated by carriage return and each column separeted by blank space. Nine cols.

    Args:
        path: the path of the csv.

    Return:
        dataset: A list. Structure: [ [ [input], [label] ], [[input], [label]] ]
    """
    dataset = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            feature = line[:8]
            label = line[8:]
            feature = [float(i) for i in feature]
            label = [float(i) for i in label]
            dataset.append([feature, label])
    return dataset

def fitness_function(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 200.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in dataset_train:
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0])**2

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)

    #p.add_reporter(neat.StdOutReporter(True))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    winner = p.run(fitness_function, 50)

    print('\nBest genome:\n{!s}'.format(winner))

    print("Output:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    error = 0.0
    for xi,xo in dataset_test:
        output = winner_net.activate(xi)
        error += (output[0]-xo[0])**2
        print("expected output {!r}, got {!r}".format(xo[0], output[0]))
    error = error/len(dataset_test)
    error = math.sqrt(error)
    print("Error: %f"%(error))

def main():
    global dataset_train, dataset_test
    local_dir = "./"
    config_path = os.path.join(local_dir, "config.txt")
    dataset_train = load_dataset(dataset_train_path)
    dataset_test = load_dataset(dataset_full_path)
    run(config_path)
    

if __name__ == '__main__':
    main()