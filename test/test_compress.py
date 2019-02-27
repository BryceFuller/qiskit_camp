# Python
import logging

# Local
import compress

# External
import pytest
from qiskit import QuantumRegister, QuantumCircuit


def test_compression_model():
    # Create a simple circuit that applies one simple flip.
    q0 = QuantumRegister(1, 'q0')
    circuit = QuantumCircuit(q0)

    logging.critical(type(circuit))

    final_circuit = compress.swap_test_with_compression_model(circuit)
    final_circuit.draw()
