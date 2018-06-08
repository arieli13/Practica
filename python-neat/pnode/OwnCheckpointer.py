from neat.reporting import BaseReporter
from neat.math_util import mean, stdev
from neat.six_util import itervalues

import gzip
import random
import time

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

from neat.population import Population
from neat.reporting import BaseReporter


class OwnCheckpointer(BaseReporter):
    def __init__(self, checkpoint_path):
        self.__checkpoint_path = checkpoint_path
        self.__generation = 0
    
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass
    
    def save_checkpoint(self, config, population, species_set):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.__checkpoint_path, self.__generation)
        print("Saving checkpoint to {0}".format(filename))
        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (self.__generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.__generation += 1
    
    def restore_checkpoint(self, filename=None):
        """Resumes the simulation from a previous saved point."""
        if filename is None:
            filename = "%s%d"%(self.__checkpoint_path, self.__generation-1)
        try:
            with gzip.open(filename) as f:
                generation, config, population, species_set, rndstate = pickle.load(f)
                random.setstate(rndstate)
                return Population(config, (population, species_set, generation))
        except Exception as e:
            print(str(e))
            return None

    def post_evaluate(self, config, population, species, best_genome):
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