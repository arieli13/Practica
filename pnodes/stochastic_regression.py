from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def load_datasets():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open("./datasets/normalized/pnode0.csv") as f:
        lines = f.readlines()[1:]
        train_lines = lines[:500]
        test_lines = lines[:]
        for line in train_lines:
            line = [float(i) for i in line.split(",")]
            x = line[:8]
            y = line[8]
            train_x.append(x)
            train_y.append(y)
        for line in test_lines:
            line = [float(i) for i in line.split(",")]
            x = line[:8]
            y = line[8]
            test_x.append(x)
            test_y.append(y)
    return train_x, train_y, test_x, test_y
    
train_x, train_y, test_x, test_y = load_datasets()

robust = SGDRegressor(loss='squared_loss',
                      penalty='l1', 
                      max_iter=2, 
                      shuffle=False, 
                      verbose=1)
robust.fit(train_x, train_y)
with open("./predictions_robust.csv", "w+") as f:
    f.write("iteration,predicted,label\n")
    for x in range(len(test_x)):
        output = robust.predict([test_x[x]])
        f.write( "%d,%f,%f\n"%(x, output, test_y[x]) )
