import qiskit
from multideterminant_prep import PrepareMultiDeterminantState as pmds

in_strings = ["0110", "0011"]
in_weights = [4,7]
circuit = pmds(in_weights, in_strings, mode='noancilla')
print(circuit.depth())
