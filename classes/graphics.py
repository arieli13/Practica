"""Plots a csv file"""
import matplotlib.pyplot as plt


def separate_csv(path_file, separator):
    """Join each column of the csv in a single list each one."""
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


def plot_csv(path_file, separator, x_index, y_indexs, x_label, y_label, title,
             printable_lines):
    """Plot the csv file.
    
    Args:
        path_file: Path of the csv file.
        separator: Separator of each column of the csv file.
        x_index: Int. The index of the x-axis of the function.
        y_indexs: A list with the number of  the cols that want
                  to display.
        x_label: Label for the x-axis of the function.
        y_label: Label for the y-axis of the function.
        title: Title for the function.
        printable_lines: A list with the type of line for each function.
                         Example: ["g+", "r-"]
    """
    header, lines = separate_csv(path_file, separator)
    for i in range(len(lines)):
        if i in y_indexs:
            plt.plot(lines[x_index], lines[i], printable_lines.pop(),
                     label=header[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    plt.close()
