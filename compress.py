# Python
import logging
from copy import deepcopy
from functools import partial
from contextlib import contextmanager
import time

from multideterminant_prep import PrepareMultiDeterminantState as pmds


# Local

# External
import pandas as pd
import numpy as np
import xarray as xr
import xyzpy as xy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, tools
#from qiskit.aqua.components.optimizers import SPSA, SLSQP
from qiskit.aqua.components.optimizers import SLSQP
from custom_spsa import SPSA



def build_model_from_parameters(circuit, theta):
    layers, num_qubits, rotations = theta.shape
    assert rotations == 3, "Expected a 3-sized vector for each theta."

    # Combine all the qubits in the circuit into one flat list
    all_qubits = [(reg, qubit) for register in circuit.qregs for reg, qubit in register]
    assert len(all_qubits) == num_qubits, \
        "There are {} qubits but expected same as parameter's shape ({})!".format(len(all_qubits), num_qubits)

    for l in range(layers):
        # Add layer of (pseudo)random single qubit rotations
        for n, (reg, qubit_idx) in enumerate(all_qubits):
            circuit.u3(theta[l][n][0], theta[l][n][1], theta[l][n][2], reg[qubit_idx])

        # Pattern with 2-qubit gates
        # change the step size in range to affect density of cx layer
        for n in range(0, num_qubits, 2):
            if (n + 1) + (l % 2) < num_qubits:  # Out-of-bounds check
                reg_1, qubit_idx_1 = all_qubits[n + (l % 2)]
                reg_2, qubit_idx_2 = all_qubits[(n + 1) + (l % 2)]
                circuit.cx(reg_1[qubit_idx_1], reg_2[qubit_idx_2])


def build_compression_model(registers, model_parameters):
    """ Given a set of input registers, builds a parametrised model of random
    gate operations which seek to approximate some other (equally-sized) target
    circuit.

    Returns:
        QuantumCircuit: a circuit consisting of random operations on the given
        registers.
    """
    model_circuit = QuantumCircuit(*registers)

    # Reshape model parameters so they make sense
    n_layers = int(len(model_parameters) / 3 / model_circuit.width())
    model_parameters = np.array(model_parameters).reshape((n_layers, model_circuit.width(), 3))

    # TODO: Store some internal parametrization that can be optimised over.
    # TODO: Create random set of gate operations based on parametrization.
    build_model_from_parameters(model_circuit, model_parameters)

    return model_circuit


def swap_test_with_compression_model(target_circuit, model_parameters):
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


def compute_approximation_fidelity(target_circuit, backend='qasm_simulator', n_shots=1000, results_fidelity_list=[], model_parameters=[]):
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
    fidelity = result_counts.get('1', n_shots) / n_shots

    results_fidelity_list.append(fidelity)
    logging.critical("Fidelity: {}".format(fidelity))

    return fidelity


def cross_validate_qnn_depth(target_circuit, n_shots, n_iters, n_layers, run=0):
    """ Runs a single cross-validation experiment with the given parameters.

    Returns:
        xr.Dataset: an xarray Dataset consisting of any number of DataArrays of
        data regarding the results of this experiment. This Dataset will be merged
        with all other experiment datasets, so in theory every experiment
        should return a fairly consistent set of data.
    """

    # TODO: Compile the circuit to determine a measure of optimised circuit depth
    # tools.compiler.compile(target_circuit, backend)
    # logging.critical("Circuit depth (compiled): {}".format(compiled_depth))


    initial_point = []

    for layer in range(1,n_layers+1):
	    # Configuration
	    layer=3
	    n_params = target_circuit.width() * layer * 3
	    variable_bounds_single = (0., 2*np.pi)
	    variable_bounds = [variable_bounds_single] * n_params

	    if len(initial_point) < n_params:
	    	#Do a quick check to make sure the number of existing params makes sense
	    	if ((n_params - len(initial_point)) % (3*target_circuit.width())) != 0:
	    		raise Exception("Unexpected number of parameters encountered") 
	    	initial_point = initial_point + (np.random.uniform(low=variable_bounds_single[0],
	                                      high=variable_bounds_single[1],
	                                      size=(n_params - len(initial_point),)).tolist())

	    print("LEN INITIAL POINT:" + str(len(initial_point)))
	    print(len(variable_bounds))
	    # Build variable bounds
	    
	   
	    # logging.critical("Initial point: {}".format(initial_point))

	    # Store resulting information
	    results_fidelity_list = []

	    # Partially define the objective function
	    maximisation_function = partial(compute_approximation_fidelity, target_circuit, "qasm_simulator", n_shots, results_fidelity_list)
	    minimization_function = lambda params: -maximisation_function(params)

	    # Call the optimiser
	    optimizer = SPSA(max_trials=n_iters, save_steps=1)
	    

	    result = optimizer.optimize(n_params, minimization_function,
	                                variable_bounds=variable_bounds, initial_point=initial_point)

	    last_params, last_score, _ = result

	    logging.critical("FINAL SCORE: {}".format(-last_score))
	    logging.critical("FINAL PARAMS: {}".format(len(last_params)))

	    # Ignore the first set of fidelities (calibration) and very last one (equal to last_score)
	    results_fidelity_list = results_fidelity_list[-((n_iters * 2) + 1):-1]

    # TODO calculate compiled depth (for simulator this is always 2*l)

    # Output results
    print("FLAAAG")
    return xr.Dataset({
        "fidelity": xr.DataArray(np.array(results_fidelity_list).reshape((n_iters, 2)), coords={"iteration": range(n_iters), "plusminus": range(2)}, dims=["iteration", "plusminus"]),
        "last_theta": xr.DataArray(np.array(last_params).reshape((n_layers, target_circuit.width(), 3)), coords={"layer": range(n_layers), "qubit": range(target_circuit.width()), "angle": ["theta", "phi", "lambda"]}, dims=["layer", "qubit", "angle"]),
        "uncompiled_target_depth": xr.DataArray(target_circuit.depth()),
        # "compiled_target_depth": xr.DataArray(compiled_depth),
        "uncompiled_model_depth": xr.DataArray(2 * n_layers),
    })


@contextmanager
def experiment_crop(fn, experiment_name):
    """ Defines how to run a crop of experiments (i.e. all the experiments in
    the grid) in parallel and store results in a file.
    """
    experiment_runner = xy.Runner(fn, var_names=None)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "experiment_results/{}_{}.h5".format(experiment_name, timestr)
    logging.critical("Running experiments: {}".format(filename))
    experiment_harvester = xy.Harvester(experiment_runner, data_name=filename)
    experiment = experiment_harvester.Crop()

    yield experiment

    #replace parallel=True with num_workers=# 
    #...to keep stop my computer from hurting itself
    #experiment.grow_missing(parallel=True)
    #experiment.grow_missing(num_workers=5)
    experiment.grow_missing()

    results = experiment.reap(wait=True, overwrite=True)
    logging.critical(results)
    logging.critical("Saved experiments: {}".format(filename))


if __name__ == "__main__":
    """
    Example which defines a circuit and then runs a set of experiments, defined
    by the grid_search parameters, across a parallel set of processes.
    """

    logging.critical("Creating the circuit...")
    in_strings = ["01", "10"]
    in_weights = [4, 7]
    target_circuit = pmds(in_weights, in_strings, mode='noancilla')
    logging.critical("Circuit depth (uncompiled): {}".format(target_circuit.depth()))

    logging.critical("Running the experiments...")
    with experiment_crop(cross_validate_qnn_depth, "experiments") as experiment:
        grid_search = {
            'n_shots': [100],
            'n_iters': [10],
            'n_layers': [3],

            'run': range(50),
        }
        constants = {
            'target_circuit': target_circuit
        }
        experiment.sow_combos(grid_search, constants=constants)
