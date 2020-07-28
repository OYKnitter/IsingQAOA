import numpy as np
import matplotlib.pyplot as plt

with open('data.txt') as file:
    data = file.read().split('\n')
    transverse = np.zeros(len(data))
    variance = np.zeros(len(data))
    rel_error = np.zeros(len(data))
    counter = 0
    for i in data:
        datum = i.split()
        transverse[counter] = float(datum[0])
        variance[counter] = float(datum[1])
        rel_error[counter] = abs((float(datum[2])-float(datum[3]))/float(datum[3]))
        counter += 1
    
    plt.plot(transverse,variance)
    plt.xlabel('Transverse Coefficient')
    plt.ylabel('Variance')
    plt.savefig('transverseVariance.png')

    plt.clf()
    plt.plot(transverse, rel_error)
    plt.xlabel('Transverse Coefficient')
    plt.ylabel('Relative Error')
    plt.savefig('transverseError.png')