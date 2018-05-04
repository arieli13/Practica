#!/bin/bash
REMOVE_LOGS='
import shutil
shutil.rmtree("./'$1'/logs")'
TXT='
registers = []
with open("./'$1'/testLog.csv", "r") as csv:
        with open("./'$1'/testLog.txt", "w") as txt:
            string = ""
            csv.readline()
            for i in csv:
                i = i.split(";")
                registers.append(i)
            registers = sorted(registers, key= lambda item: [item[6],item[0]])
            for i in registers:
                string+= "%8.6f Average test cost\n\n\tAverage training cost:\t%f\n\tHidden Layer(s):\t\t%s\n\tTotal weights:\t\t\t%d\n\tIterations:\t\t\t\t%d\n\tBatch Size:\t\t\t\t%d\n\tTraining data:\t\t\t%d\n\tTest data:\t\t\t\t%d\n\tLearning rate:\t\t\t%f\n\tTraining time:\t\t\t%f seconds\n\tTest time:\t\t\t\t%f seconds\n\n\n"%(float(i[0]), float(i[1]), i[2], int(i[3]), int(i[4]), int(i[5]), int(i[6]), int(i[7]), float(i[8]), float(i[9]), float(i[10]))
            txt.write(string)
            txt.close()'
python -c "$REMOVE_LOGS"
python "./"$1"/prediction.py"
python -c "$TXT"
tensorboard --logdir=./$1/logs/
