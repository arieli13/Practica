import tensorflow as tf

class PersistanceManager:

    def __init__(self, sess, variables, path_load, path_save=None):
        self.__sess = sess
        self.__variables = variables
        self.__saver = tf.train.Saver(variables)
        self.__ckpt = 0
        self.__path_load = path_load
        if path_save is None:
            self.__path_save = path_load
    
    def save_variables(self):
        """Save the current session...
        """
        self.__saver.save(self.__sess, self.__path_save, global_step=self.__ckpt)  # Saves the weights of the model
        self.__ckpt += 1
    
    def load_variables(self):
        """Load the current session from a saved one...
        """
        ckpt = 0
        success = True
        try:
            latest_checkpoint = tf.train.latest_checkpoint(self.__path_load)
            self.__saver.restore(self.__sess, latest_checkpoint)
            ckpt = int(latest_checkpoint.split("-")[-1])+1
            print("Model loaded successfully")
        except:
            success = False
            print("Could not load the model")
        finally:
            self.__ckpt = ckpt
            return success
