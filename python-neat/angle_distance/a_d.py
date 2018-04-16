import os
import neat
import math

#import visualize
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

dataset_path = "./datasets/dataset_input"
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
            feature = line[:3]
            label = line[3:]
            feature = [float(i) for i in feature]
            label = [float(i) for i in label]
            dataset.append([feature, label])
    return dataset

def fitness_function(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 1000*5.84
        distance_difference = 0
        angle_difference = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in dataset_train:
            output = net.activate(xi)
            distance_difference += (output[0]-xo[0])**2
            angle_difference += (output[1]-xo[1])**2
        error = distance_difference + angle_difference
        error /= (len(genomes)*2)
        genome.fitness -= error

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
    distance_difference = 0
    angle_difference = 0
    for xi, xo in dataset_test[:200]:
        output = winner_net.activate(xi)
        distance_difference += (output[0]-xo[0])**2
        angle_difference += (output[1]-xo[1])**2

        #print("expected output ({!r}, {!r}), got ({!r}, {!r})".format(xo[0],xo[1], output[0], output[1]))
    error = distance_difference + angle_difference
    error = error/400
    error = math.sqrt(error)
    print("Error: %f"%(error))

def main():
    global dataset_train, dataset_test
    local_dir = "./"
    config_path = os.path.join(local_dir, "config.txt")
    dataset_train = load_dataset(dataset_full_path)
    dataset_test = dataset_train[1000:]
    dataset_train = dataset_train[:1000]
    run(config_path)
    

if __name__ == '__main__':
    main()