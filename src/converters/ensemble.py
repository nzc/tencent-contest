import argparse, csv, sys, pickle, collections, math

def logistic_func(x):
    return 1/(1+math.exp(-x))

def inv_logistic_func(x):
    return math.log(x/(1-x))

path_1 = sys.argv[1]
path_2 = sys.argv[2]
path_3 = sys.argv[3]
weight_1 = float(sys.argv[4])
weight_2 = float(sys.argv[5])
data_1 = []
data_2 = []

f = open(path_1,'rb')
for line in f:
    data_1.append(float(line))
f.close()

f = open(path_2,'rb')
for line in f:
    data_2.append(float(line))
f.close()

assert len(data_1) == len(data_2)

f = open(path_3,'wb')
for i, d in enumerate(data_1):
    #t1 = inv_logistic_func(d)
    #t2 = inv_logistic_func(data_2[i])
    #val = logistic_func(t1*weight_1+t2*weight_2)
    val = (d*weight_1+data_2[i]*weight_2)
    f.write('%.6f'%(val)+'\n')
f.close()
