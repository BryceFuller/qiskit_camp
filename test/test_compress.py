# Python
import logging

# Local
import compress

# External
import pytest
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit


def test_compression_model():
    # Create a simple circuit that applies one simple flip.
    q0 = QuantumRegister(1, 'q0')
    circuit = QuantumCircuit(q0)

    fidelity = compress.compute_approximation_fidelity(circuit)
    assert fidelity == 1.0, "Fidelity should be 100% for the same circuit!"

    circuit.h(q0)
    fidelity = compress.compute_approximation_fidelity(circuit)
    assert np.isclose(fidelity, 0.75, rtol=1e-1)  # XXX: Very loose test :3
