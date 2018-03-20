entrenamientos_features = []
entrenamientos_labels = []

'''
with open('distancia.txt', 'a') as x:
	with open('angulo.txt', 'a') as y:
		with open('angulo_act.txt', 'a') as z:
			with open('./Entrenamientos/leftArmMovement.txt', 'r') as file: #
				for line in file:
					line = line.split(" ")
					line = map(float, line)
					x.write(str(line[0])+"\n")
					y.write(str(line[1])+"\n")
					z.write(str(line[2])+"\n")
'''
def function(file, newFile):
	with open(newFile, 'a') as x:
		with open(file, 'r') as file: #
			data = file.readlines()
			data = data[500:]
			for line in data:
				x.write(line)
			file.close()
		x.close()
