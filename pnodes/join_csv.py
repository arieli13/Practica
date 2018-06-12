import sys
sys.path.append("../classes")
import Graphics as gf

csvs_numbers = [454, 456, 458]
header = "Iteración,Memoria 10,Memoria 50,Memoria 100\n"
csvs = []
index_to_join = 2

name = ["Entrenamiento", "Validación"]
name = name[index_to_join-1]

steps = 25
lr = "1"

for i in csvs_numbers:
    csvs.append( "./tests/errors/error_log_%d.csv"%(i) )

files = []
finish_csv = []

for i in csvs:
    x = open(i, "r")
    files.append(x.readlines()[1:])
    x.close()


for i in range(len(files[0])):
    new_data = ["%d"%(i)]
    for f in files:
        new_data.append(str(float(f[i].split(",")[index_to_join])))
    new_line = ",".join(new_data)+"\n"
    finish_csv.append(new_line)
finish_csv = "".join(finish_csv)

new_path = "x.csv"
new_file = open(new_path, "w+")
new_file.write(header)
new_file.write(finish_csv)
new_file.close()
save_path = "./tests/joined_graphs/Errors/minibatch_memoria-25-validacion.png"
gf.plot_csv(new_path, ",", 0,[1, 2, 3], "Iteración", "Error cuadrático medio", "%s con %d pasos por iteración"%(name, steps), ["g.", "r.", "k.", "b."], True, save_path)

