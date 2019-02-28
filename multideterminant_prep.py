#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.quantum_info import Pauli, state_fidelity, basis_state, process_fidelity
import qiskit.aqua
import qiskit.aqua.utils.mct
import math
import numpy as np


# In[16]:


def PrepareMultiDeterminantState(weights, determinants, normalize = True, mode = 'basic'):
    '''
        Constructs multideterminant state : a linear superposition of many fermionic determinants.
        Appropriate for us as an input state for VQE.
        
        Input: np.array of REAL weights, and list of determinants. 
                Each determinant is a bitstring of length n.
                
        Output: a QuantumCircuit object.
                If mode == basic, acting on 2n qubits (n logical, n ancillas).
                Otherwise, acting on n+1 qubits.

                The depth of the circuit goes like O(n L) where L = len(weights), in basic mode,
                and exponentially in n when only 1 ancilla.
                
        This is an implementation of the circuit in https://arxiv.org/abs/1809.05523
    '''
    # number of determinants in the state
    L = len(weights)
    if L != len(determinants):
        raise Exception('# weights != # determinants')
        
    # Check normalization
    norm = np.linalg.norm(weights)
    #print('Norm is: {}'.format(norm))
    if normalize:
        weights /= norm
    elif abs(norm-1) > 10**-5:
        raise Exception('Weights do not produce a normalized vector.')
        
    #TODO: check that weights are normalized
    
    # number of orbitals that determinants can be made of
    n = len(determinants[0])
    
    # the first n qubits will be used for orbitals, the next as the controlled rotation ancilla, 
    # the last n as ancillas will be used for the n-Toffoli

    qubits = QuantumRegister(n, 'q')
    cqub = QuantumRegister(1, 'c')
    
    if mode == 'basic':
        ancillas = QuantumRegister(n-1, 'a')
        registers = [qubits, cqub, ancillas]
    else:
        # no ancillas
        registers = [qubits, cqub]
        
    # initialize
    circ = QuantumCircuit(*registers)
    for i in range(n):
        if determinants[0][i] == 1:
            circ.x(qubits[i])
    circ.x(cqub)
    b = 1 # the beta
    a = 0
    
    # iterate over all determinants that must be in the final state
    
    for step in range(L-1):
        # choose first qubit on which D_l and D_{l+1} differ
        old_det = determinants[step]
        new_det = determinants[step+1]
        a = weights[step]
        
        # Find first bit of difference
        different_qubit = 0
        while different_qubit <= n:
            if old_det[different_qubit] != new_det[different_qubit]:
                break
            else:
                different_qubit +=1
        if different_qubit == n:
            raise Exception('Determinants {} and {} are the same.'.format(old_det, new_det))
            
        # Compute the rotation angle
        # Equation is cos(g) beta_l = alpha_l
        angle = math.acos(a/b)
        #print('Step {} angle is {}'.format(step, angle))

        # beta_{l+1} is
        b = b * math.sin(angle)
        
        if old_det[different_qubit] == 1:
            b = -b
            #print('Flipped beta sign.')
        
        
        '''
        want a controlled-Y rotation, but can do controlled-Z, so map Y basis to Z basis. 
        1) first map Y to X (with U1(-pi/2) gate)
        2) X to Z, with Hadamard
        3) apply z-rotation
        4) undo this
        ''' 
        #circ.u1(-math.pi/2, qubits[different_qubit])
        #circ.h(qubits[different_qubit])
        #circ.crz(angle, cqub, qubits[different_qubit])
        #circ.h(qubits[different_qubit])
        #circ.u1(-math.pi/2, qubits[different_qubit])
        circ.cu3(2 * angle,0,0, cqub, qubits[different_qubit])
        

        # Now must do an n-qubit Toffoli to change the |1> to |0>
        # but controlled on the |D_l> bitstring
        flip_all(circ, qubits, old_det)
        if mode != 'basic':
            # no ancillas
            circ.mct(qubits, cqub, None, mode = 'noancillas')
        else:
            #circ.mct(qubits, cqub, ancillas, mode = 'basic')
            nToffoliAncillas(circ, qubits, ancillas, cqub)
        flip_all(circ, qubits, old_det) # undo
        
        #if step > 0:
        #    break
        
        # If b not same sign as weight, flip sign
        #if b * weights[step+1]:
        #    circ.z(cqub)
        #    b = -b
        
        # Now continue flipping the rest of the bits
        for i in range(different_qubit+1, n):
            if old_det[i] != new_det[i]:
                circ.cx(cqub, qubits[i])
        circ.barrier()
    
    # Finally check that the sign of the last weight is correct
    # and set the ancilla to zero. 
    if b * weights[-1] < 0:
        # must flip sign
        circ.z(cqub)
    # Remove the |1>
    flip_all(circ, qubits, new_det)
    nToffoliAncillas(circ, qubits, ancillas, cqub)
    flip_all(circ, qubits, new_det) # undo    
        
    return circ


# In[3]:


def flip_all(circ, tgt, bits):
    '''
        Flips the qubits of tgt where the bit in the bitstring is 0.
    '''
    for i in range(len(bits)):
        if bits[i] == 0:
            # flip
            circ.x(tgt[i])


# In[4]:


def nToffoliAncillas(circ, ctrl, anc, tgt):
    '''
        Returns a circuit that implements an n-qubit Toffoli using n ancillas.
    '''

    n = len(ctrl)
    
    # compute
    circ.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, n):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])

    # copy
    circ.cx(anc[n-2], tgt[0])

    # uncompute
    for i in range(n-1, 1, -1):
        circ.ccx(ctrl[i], anc[i-2], anc[i-1])
    circ.ccx(ctrl[0], ctrl[1], anc[0])    
    return circ

#from qiskit.tools.visualization import circuit_drawer
#circuit_drawer(circ)


# In[18]:


'''weights3 = np.array([1, -1, 2])
weights3 = weights3 / np.linalg.norm(weights3)
dets3 = [ (1,0,1), (1,1,0), (0,1,1)] # (5, 6, 3)
print(weights3)
def dec(tup):
    n = len(tup)
    tot = 0
    for i in range(n):
        tot += tup[i] * 2**(n-i-1)
    return tot

print([dec(tup) for tup in dets3])
c, _, _, _ = MDCirc(weights3, dets3, mode = 'n')

job = execute(c, sim).result()
vec = job.get_statevector(c)
# dets3 = [(1,1,0), (1,0,1), (0,1,1)]
for i in range(len(vec)):
    if abs(vec[i]) > 10**-4:
        print('{}: {}'.format(i, vec[i]))
'''


# In[ ]:




