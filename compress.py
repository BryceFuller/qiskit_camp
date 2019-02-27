# Python
import logging

# Local

# External
from qiskit import QuantumCircuit
from qiskit import execute


def build_compression_model(circuit):
    """ Given a circuit, builds a parametrised sub-circuit that runs in
    parallel with and approximately models (with compression) the original
    circuit.

    Return:
        QuantumCircuit: new combined circuit consisting of the original circuit
        and the model circuit.
    """

    # We assume that all QUANTUM registers of the original circuit are to be
    # modelled.
    new_registers = [reg.__class__(reg.size, '{}_model'.format(reg.name))
                     for reg in circuit.qregs]
    model_circuit = QuantumCircuit(*new_registers)

    # TODO: Build the model's compression circuit here.

    # Append the two circuits together.
    top_circuit = circuit + model_circuit

    # Synchronisation barrier just so we know where the original circuits ends.
    top_circuit.barrier()

    # Perform a SWAP test here.
    # TODO: This should be done outside of this method.
    # top_circuit.swap(tuple(circuit.qregs), tuple(new_registers))

    return top_circuit
