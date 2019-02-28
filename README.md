# qiskit_camp
## State preparation via Quantum compression 

### Usage:

See the compress.py's __main__ block for an example experimental run.


#### cross_validate_qnn_depth(target_circuit, n_shots, n_iters, n_layers, run=0):

inputs: 
    target_circuit: QuantumCircuit object encoding your state preparation
    n_shots: integer number of samples used when evaluating training circuit on one set of parameters
    n_iter:  integer number of SPSA optimization steps (number of parameter updates)
    n_layers: the number of layers in the learned quantum circuit. One layer has 
              a tiling of single qubit rotations followed by a tiling of two qubit 
              entangling operations. 
    n_runs: number of simulations to run in parrallel
              
 returns:
     xr.Dataset object encoding the parameters and fidelities for various QNN circuit depths


#### def build_compression_model(registers, model_parameters):

inputs: 
    registers: QuantumRegisters to be used in compressed circuit
    model_parameters: ndarray of parameters for learned model (obtained from cross_validate_qnn_depth)
    
returns
    compressed_circuit: QuantumCircuit object
    


\
