import tensorflow as tf
import time
pnode_number = 0

def read_datasets(path, training_registers, skip):
    train = []
    validation = []
    test = []
    with open(path) as f:
        lines = f.readlines()[skip:]
        lines =  [ [[y[:8]], [y[8:]]] for y in [[float(k) for k in i.split(",")] for i in lines ]]
        print(lines[0])
        train = lines[:training_registers]
        validation = lines[training_registers:]
        test = lines[:]
    return train, validation, test

        

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    train, val, test, train_i, val_i, test_i = create_datasets("./datasets/normalized/pnode%d.csv"%(pnode_number), 500, 1)
    sess.run([train_i, val_i, test_i])
    cont = 0
    start = time.time()
    try:
        while True:
            sess.run(test.get_next())
            cont += 1
    except tf.errors.OutOfRangeError:
        print("%f"%(time.time()-start))