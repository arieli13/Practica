import os
import neat
import math

#import visualize
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

dataset_path = "../../datasets_angle_distance/cart_leftArmMovement.csv"

dataset_train = []
dataset_test = []

train_registers = 494

log = [] #Saves mean and stdv of each generation

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
        lines = f.readlines()[1:]
        for line in lines:
            line = line.split(",")
            feature = line[:4]
            label = line[4:]
            feature = [float(i) for i in feature]
            label = [float(i) for i in label]
            dataset.append([feature, label])
    return dataset


def test(winner_net):
    error = 0.0
    distance_difference = 0.0
    angle_difference = 0.0
    global dataset_test
    log_predictions = []
    #dataset_test = dataset_test[:200]
    for xi, xo in dataset_test:
        output = winner_net.activate(xi)
        distance_difference += (output[0]-xo[0])**2
        angle_difference += (output[1]-xo[1])**2
        log_predictions.append( "{!r};{!r};{!r};{!r}\n".format(xo[0],xo[1], output[0], output[1]) )
        #print()
    error = distance_difference + angle_difference
    error = error/(2*len(dataset_test))
    error = math.sqrt(error)
    print("RMSE: %f"%(error))
    with open("./predictions_log.csv", "w+") as f:
        f.write("".join(log_predictions))

def fitness_function(genomes, config):
    
    #mean_error = 0 #mean error of generation
    #sum_sc_error = 0
    #sum_error = 0
    #gen_size = len(genomes)

    #min_fitness = 10000
    #max_fitness = -10000

    for genome_id, genome in genomes:
        distance_difference = 0
        angle_difference = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in dataset_train:
            output = net.activate(xi)
            distance_difference += (output[0]-xo[0])**2
            angle_difference += (output[1]-xo[1])**2
        error = distance_difference + angle_difference
        error /= (train_registers*2)
        error = math.sqrt(error)

        #if min_fitness > error:
            #min_fitness = error
        #elif max_fitness < error:
            #max_fitness = error
        
        #mean_error += error
        #sum_sc_error += error*error

        genome.fitness = error*-1 #max

    """"sum_error = mean_error
    mean_error /= gen_size
    stdv_error = sum_sc_error + -2*mean_error*sum_error + gen_size*mean_error*mean_error
    stdv_error /= (gen_size-1)
    stdv_error = math.sqrt(stdv_error)

    log_data = {}
    log_data["mean"] = mean_error
    log_data["stdv"] = stdv_error
    log_data["best"] = min_fitness
    log_data["worst"] = max_fitness

    log.append(log_data)"""


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)

    winner = p.run(fitness_function, 7000)

    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    test(winner_net)

def save_logs():
    global log
    cont = 0
    with open("./log_poblation.csv", "w+") as f:
        for l in log:
            f.write("%d;%f;%f;%f;%f\n"%(cont, l["mean"], l["stdv"], l["best"], l["worst"]))
            cont += 1

def main():
    global dataset_train, dataset_test
    local_dir = "./"
    config_path = os.path.join(local_dir, "config.txt")
    dataset_train = load_dataset(dataset_path)
    #dataset_test = load_dataset("./datasets/dataset.txt")
    dataset_test = dataset_train[:]
    dataset_train = dataset_train[:train_registers]
    run(config_path)
    #save_logs()
    

if __name__ == '__main__':
    main()