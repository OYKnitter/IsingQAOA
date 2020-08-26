import numpy as np
import netket as nk


class SKSpinGlassEnergy:
    def __init__(self, cf):
        self.cf = cf


    def _construct_ising_hamiltonian(self, J, offset=0):
        # Edwards-Anderson model
        N = J.shape[0]
        # Pauli x Matrix
        sx = [[0., 1.],[1., 0.]]
        # Pauli z Matrix
        sz = [[1., 0.], [0., -1.]]
        # create graph
        edges = []
        for i in range(N):
            for j in range(i, N):
                if J[i,j] != 0.: edges.append([i, j])
        g = nk.graph.CustomGraph(edges)
        # system with spin-1/2 particles
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        ha = nk.operator.LocalOperator(hi, offset)
        for i in range(N):
            for j in range(N):
                if J[i,j] != 0.:
                    if i != j:
                        ha += -J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
                    else: #Transverse field component -- coefficients encoded in diagonal of J matrix
                        ha += -J[i,j] * nk.operator.LocalOperator(hi, sx, [i])
        # ha.to_dense() for matrix representation of ha
        return ha, g, hi


    def laplacian_to_hamiltonian(self, J):
        hamiltonian, graph, hilbert = self._construct_ising_hamiltonian(J)
        return hamiltonian, graph, hilbert