import math

with open("./datasets/leftArmMovement.txt", "r") as f:
    with open("./x.txt", "w+") as n_f:
        lines = f.readlines()
        new_file = ""
        cont = 0
        for line in lines:
            line = [float(x) for x in line.split(" ") if x]
            y = math.cos( line[1] )*line[0]
            x = math.sin( line[1] )*line[0]

            yp = math.sin(line[2])*0.05
            xp = math.cos(line[2])*0.05

            line = [ str(x), str(y), str(xp), str(yp), str(x+xp), str(y+yp) ]
            line = " ".join([str(x) for x in line]) + "\n"
            new_file += line
        n_f.write(new_file)
        print cont
