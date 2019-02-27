import qiskit
import random as r

# External
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute

#Theta is an l-tuple where each entry is a n-size array
# where n is the # of input qubits

seed = 123456789
r.seed(seed)

def main(theta, num_qubits, layers):
	qreg = QuantumRegister(num_qubits, "q_mod")
	creg = ClassicalRegister(num_qubits, "c_mod")
	model = QuantumCircuit(qreg, creg)
	

	funcs = [model.rx, model.ry, model.rz]

	for l in range(layers):
		
		#Add layer of (pseudo)random gates
		for n in range(num_qubits):
			ri = r.randint(0,2)
			f = funcs[ri]
			print(l,n)
			f(theta[l][n], qreg[n])
		
		model.barrier()	
		#Naive 2-qubit gates
		#change the step size in range to affect density of cx layer
		for n in range(0,num_qubits,2):
			if n+1+l%2 < num_qubits:
				model.cx(qreg[n+l%2],qreg[n+1+l%2])
				pass
		model.barrier()
	return model

####################
#\\\\\\\\\\\\\\\\\\\
####################

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
print(theta)
model = main(theta, n, l)
    
print(model)
# theta = tuple()

