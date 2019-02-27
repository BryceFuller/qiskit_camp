# Python
import logging

# Local
import compress

# External
import pytest
from qiskit import QuantumRegister, QuantumCircuit


def test_create_model():
    # Create a simple circuit that applies one simple flip.
    q0 = QuantumRegister(1, 'q0')
    circuit = QuantumCircuit(q0)

    logging.critical(type(circuit))

    final_circuit = compress.build_compression_model(circuit)
    final_circuit.draw()
