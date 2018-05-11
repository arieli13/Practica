"""PersistanceManager class. Saves and loads variables of Tensorflow."""
import tensorflow as tf


class PersistanceManager:
    """Save and load variables of Tensorflow."""

    def __init__(self, sess, variables, path_load, path_save=None):
        """Initialize all variables.

        Args:
            sess: Instance of tf.Session(). The session where the
                  variables have been created.
            variables: A dictionary with the variables to store/load.
            path_load: Path where the variables will be loaded.
            path_save: Path where the variables will be saved.
                       If None, it adquires the same value of path_load.
        """
        self.__sess = sess
        self.__variables = variables
        self.__saver = tf.train.Saver(variables)
        self.__ckpt = 0
        self.__path_load = path_load
        if path_save is None:
            self.__path_save = path_load
    
    def save_variables(self):
        """Save the specified variables of the session."""
        self.__saver.save(self.__sess, self.__path_save,
                          global_step=self.__ckpt)  # Saves the weights of the model
        self.__ckpt += 1
    
    def load_variables(self):
        """Load the current session from a saved one."""
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
