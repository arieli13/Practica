import matplotlib.pyplot as plt

def separate_csv(path_file, separator):
    lines = []
    with open(path_file, "r") as f:
        lines = f.readlines()

    header = lines[0].split(separator)
    lines = lines[1:]
    cols = [[]for i in range(len(header))]
    lines = list(map(lambda line: [float(i) for i in line.split(separator) if i], lines))
    for line in lines:
        for i in range(len(line)):
            cols[i].append(line[i])
    return header, cols

def plot_csv(path_file, separator, x_index, y_indexs, x_label, y_label, title, printable_lines):

    header, lines = separate_csv(path_file, separator)

    for i in range(len(lines)):
        if i in y_indexs:
            plt.plot(lines[x_index], lines[i], printable_lines.pop(), label=header[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()

    plt.close()

