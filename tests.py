import model as m
#import numpy as np
import csv
import pandas as pd

x = []
y = []
with open("test.csv", "r") as file:
    read = csv.reader(file)
    for line in read:
        if line[0] == 'pMean':
            continue
        xrow = []
        y.append([float(line[0])])
        xrow.append(1)
        xrow.append(float(line[3]))
        xrow.append(float(line[1]))
        xrow.append(float(line[2]))
        x.append(xrow)

df = pd.read_csv("test.csv")

ys = df.pMean
print(ys)
xs = df[["elevation", "lon", "lat"]]
print(xs)

mod = m.lm_model()


mod.make(y, x)

mod.print_betaHat()

#mod.std_error()

mod.print_stdError()