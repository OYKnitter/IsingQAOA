import numpy as np
import random
import math

def read_or_write(cf):
    # 2D Lattices with Periodic Boundary Conditions and Nearest Neighbor Interactions
    # Can either generate a random example and list it in InputData.txt OR read an existing example from InputData.txt
    # Existing example data must be formatted identically to the format used by the Cologne spin glass server.
    # Data for an existing example must coincide with '--input_size' flag.
    nodes = int(cf.input_size[0])
    if cf.random_example:
        # Program accepts a node size and constructs a random example
        size = int(math.sqrt(nodes))
        print("Generating a random {}x{} 2D spin glass problem...".format(size,size))

        mu = 0
        sigma = 1

        # Randomly construct energy matrix J, 2D Spin Glass with Periodic BC
        J = np.zeros((nodes,nodes))
        with open('Input/InputData.txt','w') as file:
            file.write('name: ' + str(size) + 'x' + str(size) + ' spin glass example with periodic boundary conditions.\n\n')
            for i in range(nodes):
                value = random.gauss(mu,sigma)
                target = (i+1)%size + (i - i%size)
                J[i,target] = value
                file.write(str(i+1)+' '+str(target+1)+' '+str(value)+'\n')
    
            for i in range(nodes):
                value = random.gauss(mu,sigma)
                target = (i + size)%nodes
                J[i,target] = value
                file.write(str(i+1)+' '+str(target+1)+' '+str(value)+'\n')
    else:
        #Program reads a predetermined example from InputData.txt. Node size flug must match input data!
        print("Reading predetermined example...")
        J=np.zeros((nodes,nodes))
        with open('Input/InputData.txt') as file:
            data = file.read().split('\n')
            for i in data:
                if i.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    datum = i.split()
                    J[int(datum[0])-1,int(datum[1])-1]=float(datum[2])

    if cf.transverse !=0.0:
        for i in range(nodes):
            J[i,i] = cf.transverse

    return J