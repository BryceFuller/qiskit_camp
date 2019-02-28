# Python
import logging
from copy import deepcopy
from functools import partial

# Local

# External
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.aqua.components.optimizers import *


def build_compression_model(registers, model_parameters=None):
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
    # build_model(model_parameters)

    return model_circuit


def swap_test_with_compression_model(target_circuit, model_parameters=None):
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
    model_circuit = build_compression_model(new_registers, model_parameters)

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


def compute_approximation_fidelity(target_circuit, backend='qasm_simulator', n_shots=1000, model_parameters=None):
    """ Performs a set of runs on the target circuit and an approximation model
    to compute the approximation's fidelity.

    Returns:
        float: a measure of fidelity in [0, 1], where 1 represents the maximum
        achievable approximation of the target circuit.
    """
    final_circuit = swap_test_with_compression_model(target_circuit, model_parameters)

    # Execute the SWAP test circuit
    simulator = Aer.get_backend(backend)
    job = execute(final_circuit, backend=simulator, shots=n_shots)

    # Return a measurement of fidelity
    result_counts = job.result().get_counts(final_circuit)
    fidelity = result_counts.get('1', result_counts.get('0')) / sum(result_counts.values())
    return fidelity

    
def cross_validate_qnn_depth(target_circuit, min_l, max_l, stepsize=1, optimizer="SPSA", optimizer_params=None, backend='qasm_simulator', n_shots=1000):
    '''Fits many qnn's of differing depth to approximate the state produced by applying the target_circuit
    to the all zero state.
    
    Returns:
        array of dictionaries, one for each depth value tried. 
        Each dictionar is of the form: 
            {depth: integer number of parameterized layers (single qubit & two qubit rotations are one layer,)
            thetas: array of best found thetas,
            fidelity: float indicating inner product between target state and the qnn output state,
            [ADD LATER] history: array of fidelities for intermediate training steps}
    '''
    results = []
    for l in range(min_l, max_l, stepsize):
	current_instance = {}
	current_instance["layers"]
        #Run optimization routine with l-layer qnn


if __name__ == "__main__":

    q0 = QuantumRegister(1, 'q0')
    circuit = QuantumCircuit(q0)
    circuit.h(q0)

    # Number of parameters
    n_layers = 5
    n_params = circuit.width() * n_layers

    test_circuit = swap_test_with_compression_model(circuit)

    objective_function = partial(compute_approximation_fidelity, circuit, "qasm_simulator", 100)

    optimizer = SPSA(max_trials=100)
    optimizer.optimize(n_params, objective_function, variable_bounds=None, initial_point=None)
        #Transfer data into dict
        #TODO calculate compiled depth (for simulator this is always 2*l)
    	
    
    
    
    
