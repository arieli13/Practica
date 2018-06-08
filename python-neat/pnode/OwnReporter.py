from neat.reporting import BaseReporter
from neat.math_util import mean, stdev
from neat.six_util import itervalues

class OwnReporter(BaseReporter):
    def __init__(self):
        self.__gen_num = 0
        self.__fit_mean = 0
        self.__fit_stddev = 0
        self.__fit_best = 0
    
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        #print("Gen: %d\t\tMean: %f\t\tStddev: %f\tBest: %f" % (self.__gen_num, self.__fit_mean, self.__fit_stddev, self.__fit_best))
        #print("Gen: %d\tBest: %f"%(self.__gen_num, self.__fit_best))
        #self.__gen_num += 1
        pass

    def post_evaluate(self, config, population, species, best_genome):
        #fitnesses = [c.fitness for c in itervalues(population)]
        #self.__fit_mean = mean(fitnesses)
        #self.__fit_stddev = stdev(fitnesses)
        #self.__fit_best = best_genome.fitness        
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass