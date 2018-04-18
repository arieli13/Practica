with open("./datasets/leftArmMovement.txt", "r") as f:
    with open("./x.txt", "w+") as n_f:
        lines = f.readlines()
        new_file = ""
        cont = 0
        for line in lines:
            line = [float(x) for x in line.split(" ") if x]
            if line[1]<-3 and line[4]>3:
                cont += 1
            else:
                line = " ".join([str(x) for x in line]) + "\n"
                new_file += line
        n_f.write(new_file)
        print cont
