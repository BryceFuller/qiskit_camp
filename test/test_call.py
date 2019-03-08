from gen_qnn import build_model

import qiskit
import random as r

# External
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute

n = 10
l = 2

q_in = QuantumRegister(n, "q_in")
c_in = ClassicalRegister(n, "c_in")
input = QuantumCircuit(q_in, c_in)

theta = []
for i in range(l):
        theta.append(list())
        for j in range(n):
                theta[i].append(j)

model = build_model(theta, n, l)
print(model)
