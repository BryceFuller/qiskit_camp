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

    # Optimise a random compression circuit and ensure it converges to fidelity=1.0
    results = compress.cross_validate_qnn_depth(circuit, n_shots=100, n_iters=30, n_layers=3)
    assert np.isclose(results.fidelity.isel(iteration=-1).max(dim='plusminus').squeeze(), 1.0, rtol=1e-1)
