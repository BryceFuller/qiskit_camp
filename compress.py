# Python
import logging

# Local

# External
from qiskit import QuantumCircuit, ClassicalRegister


def build_compression_model(orig_circuit):
    """ Given a circuit, builds a parametrised sub-circuit that runs in
    parallel with and approximately models (with compression) the original
    circuit.

    Return:
        QuantumCircuit: new combined circuit consisting of the original circuit
        and the model circuit.
    """

    # We assume all quantum registers of the original circuit are to be
    # modelled.
    new_registers = [reg.__class__(reg.size, '{}_model'.format(reg.name))
                     for reg in orig_circuit.qregs]
    model_circuit = QuantumCircuit(*new_registers)

    # TODO: Build the model's compression circuit here.
    # model_circuit.h(*new_registers)

    # Append the two circuits together.
    top_circuit = orig_circuit + model_circuit

    # Synchronisation barrier just so we know where the original circuits ends.
    top_circuit.barrier()

    # Perform a SWAP test here on the quantum registers.
    # TODO: This should be done outside of this method.
    for orig_reg, model_reg in zip(orig_circuit.qregs, model_circuit.qregs):
        top_circuit.swap(orig_reg, model_reg)

        # Build new classical measurement registers.
        measure_reg_orig  = ClassicalRegister(orig_reg.size, '{}_measure'.format(orig_reg.name))
        measure_reg_model = ClassicalRegister(orig_reg.size, '{}_measure'.format(model_reg.name))
        top_circuit.add_register(measure_reg_orig, measure_reg_model)

        # Measure the swapped registers.
        top_circuit.measure(orig_reg, measure_reg_orig)
        top_circuit.measure(model_reg, measure_reg_model)

    return top_circuit
