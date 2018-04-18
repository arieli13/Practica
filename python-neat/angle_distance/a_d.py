import os
import neat
import math

#import visualize
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

dataset_path = "./datasets/leftArmMovement_new"
dataset_train_path = dataset_path+"_train.txt"
dataset_test_path = dataset_path+"_test.txt"
dataset_full_path = dataset_path+".txt"

dataset_train = []
dataset_test = []

train_registers = 200

log = [] #Saves mean and stdv of each generation
log_each_ind = [] #[gen1[], gen2[]...]

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


def test(winner_net):
    error = 0.0
    distance_difference = 0.0
    angle_difference = 0.0
    global dataset_test
    #dataset_test = dataset_test[:200]
    for xi, xo in dataset_test:
        output = winner_net.activate(xi)
        distance_difference += (output[0]-xo[0])**2
        angle_difference += (output[1]-xo[1])**2
        #print("expected output ({!r}, {!r}), got ({!r}, {!r})".format(xo[0],xo[1], output[0], output[1]))
    error = distance_difference + angle_difference
    error = error/(2*len(dataset_test))
    error = math.sqrt(error)
    print("RMSE: %f"%(error))

def test_normalized(winner_net):
    error = 0.0
    distance_difference = 0.0
    angle_difference = 0.0
    global dataset_test
    #dataset_test = dataset_test[:200]
    for xi, xo in dataset_test:
        output = winner_net.activate(xi)
        distance_difference += ( (output[0]*2.69) -xo[0])**2
        angle_difference += ((output[1]*math.pi*2-math.pi)-xo[1])**2
        #print("expected output ({!r}, {!r}), got ({!r}, {!r})".format(xo[0],xo[1], (output[0]*2.69), (output[1]*math.pi*2-math.pi)))
    error = distance_difference + angle_difference
    error = error/(2*len(dataset_test))
    error = math.sqrt(error)
    print("RMSE: %f"%(error))

def fitness_function(genomes, config):
    
    mean_error = 0 #mean error of generation
    sum_sc_error = 0
    sum_error = 0
    gen_size = len(genomes)

    ind_list = []

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

        ind_list.append(error) #save each genome error of the gen
        
        mean_error += error
        sum_sc_error += error*error

        genome.fitness = error*-1 #max
    
    log_each_ind.append(ind_list) #save all the genomes error each gen separated by list

    sum_error = mean_error
    mean_error /= gen_size
    stdv_error = sum_sc_error + -2*mean_error*sum_error + gen_size*mean_error*mean_error
    stdv_error /= (gen_size-1)
    stdv_error = math.sqrt(stdv_error)

    log_data = {}
    log_data["mean"] = mean_error
    log_data["stdv"] = stdv_error
    log_data["best"] = min(ind_list)
    log_data["worst"] = max(ind_list)

    log.append(log_data)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)

    #p.add_reporter(neat.StdOutReporter(True))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    winner = p.run(fitness_function, 500)

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
    gen_cont = 0
    with open("log_generation.csv", "w+") as f:
        for gen in log_each_ind:
            for ind in gen:
                f.write("%d;%f\n"%(gen_cont, ind))
            gen_cont += 1

def main():
    global dataset_train, dataset_test
    local_dir = "./"
    config_path = os.path.join(local_dir, "config.txt")
    dataset_train = load_dataset(dataset_full_path)
    #dataset_test = load_dataset("./datasets/dataset.txt")
    dataset_test = dataset_train[train_registers:]
    dataset_train = dataset_train[:train_registers]
    run(config_path)
    save_logs()
    

if __name__ == '__main__':
    main()