import numpy as np
import random
import math

def read_or_write(cf):
    # Can either generate a random example and list it in InputData.txt OR read an existing example from InputData.txt
    # Existing example data must be formatted identically to the format used by the Cologne spin glass server.
    # Each connection is expressed as a single line giving the two nodes and the weight, with one space between each value.
    # For example, a connection between nodes 1 and 7, with weight 0.592, must be formatted in a single line as either "1 7 0.592" or "7 1 0.592"
    # The weight must always be listed after the nodes. The order of the nodes does not matter, but inputting both orderings will lead to double-counting.
    # Data for an existing example must coincide with '--input_size' flag.
    # Make sure the problem type flag is set appropriately. Program will not throw exception if graph data for MaxCut is fed into the spin glass Hamiltonian generator.
    nodes = int(cf.input_size[0])
    J=np.zeros((nodes,nodes))

    # Program accepts a node size and constructs a random example
    if cf.random_example or cf.metatrain or cf.metatest:
        # Random weight distribution parameters for spin glass
        mu = 0
        sigma = 1
        #Edge density for MaxCut
        p = 0.5

        # 2D Lattices with Periodic Boundary Conditions and Nearest Neighbor Interactions
        if cf.pb_type == 'spinglass':
            size = int(math.sqrt(nodes))
            print("Generating a random {}x{} 2D spin glass problem...".format(size,size))

            # Randomly construct energy matrix J, 2D Spin Glass with Periodic BC
            with open('src/util/Input/InputData.txt','w') as file:
                file.write('name: ' + str(size) + 'x' + str(size) + ' spin glass example with periodic boundary conditions.\n\n')
                for i in range(nodes):
                    value = random.gauss(mu, sigma)
                    target = (i+1)%size + (i - i%size)
                    J[i,target] = value
                    file.write(str(i+1)+' '+str(target+1)+' '+str(value)+'\n')
    
                for i in range(nodes):
                    value = random.gauss(mu, sigma)
                    target = (i + size)%nodes
                    J[i,target] = value
                    file.write(str(i+1)+' '+str(target+1)+' '+str(value)+'\n')
        # SK spin glass model
        elif cf.pb_type == 'spinglass-SK':
            print('Generating a random size {} SK spinglass problem...'.format(nodes))

            # Randomly construct energy matrix J corresponding with weighted complete graph
            with open('src/util/Input/InputData.txt','w') as file:
                file.write('name: ' + str(nodes) + ' node Sherrington-Kirkpatrick spin glass example.\n\n')
                for i in range(nodes):
                    for j in range(i + 1, nodes):
                        value = random.gauss(mu, sigma)
                        J[i ,j] = value
                        file.write(str(i+1)+' '+str(j+1)+' '+str(value)+'\n')

        # MaxCut problem with edge density p
        elif cf.pb_type == 'maxcut':
            print('Generating a random MaxCut problem with {} nodes and edge density {}...'.format(nodes,p))
            with open('src/util/Input/InputData.txt','w') as file:
                file.write('name: ' + str(nodes) + ' node graph with edge density' + str(p) + 'for MaxCut testing.\n\n')
                for i in range(nodes):
                    for j in range(i + 1, nodes):
                        value = float(np.random.binomial(1, p))
                        J[i ,j] = value
                        file.write(str(i+1)+' '+str(j+1)+' '+str(value)+'\n')
        
        else:
            raise Exception("Unknown problem type.")
 
    else:
        # Program reads a predetermined example from InputData.txt. Node size flag must match input data!
        # Make sure the problem type flag is set appropriately.
        print("Reading predetermined example...")
        with open('src/util/Input/InputData.txt') as file:
            data = file.read().split('\n')
            for i in data:
                if i.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    datum = i.split()
                    J[int(datum[0])-1,int(datum[1])-1]=float(datum[2])

    if cf.transverse !=0.0:
        for i in range(nodes):
            J[i,i] = cf.transverse

    return J



#Find another place to put this sanity check function later.
# Ground truth solver
#from src.ising_gt import ising_ground_truth
#score, state, time_elapsed = ising_ground_truth(cf, J)
#print("The ground truth score is {}".format(score))
#print("Time elapsed: {}".format(time_elapsed))