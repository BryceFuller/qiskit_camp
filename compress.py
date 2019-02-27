# Python
import logging
from copy import deepcopy

# Local

# External
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


def build_compression_model(registers, random_seed=0):
    """ Given a set of input registers, builds a parametrised model of random
    gate operations which seek to approximate some other (equally-sized) target
    circuit.

    Returns:
        QuantumCircuit: a circuit consisting of random operations on the given
        registers.
    """
    model_circuit = QuantumCircuit(*registers)

    # TODO: Store some internal parametrization that can be optimised over.
    # TODO: Create random set of gate operations based on parametrization.

    return model_circuit


def swap_test_with_compression_model(target_circuit):
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
                     for reg in target_circuit.qregs]

    # TODO: Build the model's compression circuit here.
    model_circuit = build_compression_model(new_registers)

    # Append the two circuits together.
    top_circuit = target_circuit + model_circuit

    # Synchronisation barrier just so we know where the original circuits ends.
    top_circuit.barrier()

    # Performs a swap test between the compression model and original circuit.
    # ------------------------------------------------------------------------
    # Firstly, create an ancillary in superposition to store the swap test result.
    swap_tester = QuantumRegister(1, 'swap_tester')
    top_circuit.add_register(swap_tester)
    top_circuit.h(swap_tester)

    # Next, we perform controlled SWAPs using the swap tester.
    for orig_reg, model_reg in zip(target_circuit.qregs, model_circuit.qregs):
        for i in range(orig_reg.size):
            top_circuit.cswap(swap_tester[0], orig_reg[i], model_reg[i])

    # Finally, we re-interfere the branches and measure the swap tester.
    top_circuit.h(swap_tester)
    top_circuit.x(swap_tester)  # Make it so test measurement is P(model == target).
    swap_test_result = ClassicalRegister(1, 'swap_test_result')
    top_circuit.add_register(swap_test_result)
    top_circuit.measure(swap_tester, swap_test_result)

    return top_circuit


def compute_approximation_fidelity(target_circuit, backend='qasm_simulator', n_shots=1000):
    """ Performs a set of runs on the target circuit and an approximation model
    to compute the approximation's fidelity.

    Returns:
        float: a measure of fidelity in [0, 1], where 1 represents the maximum
        achievable approximation of the target circuit.
    """
    final_circuit = swap_test_with_compression_model(target_circuit)

    # Execute the SWAP test circuit
    simulator = Aer.get_backend(backend)
    job = execute(final_circuit, backend=simulator, shots=n_shots)

    # Return a measurement of fidelity
    result_counts = job.result().get_counts(final_circuit)
    fidelity = result_counts.get('1', result_counts.get('0')) / sum(result_counts.values())
    return fidelity
