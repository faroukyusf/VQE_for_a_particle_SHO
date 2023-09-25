#!/usr/bin/env python
# coding: utf-8

# Estimator_Expected is a function that does the quatum routine part using the estimator class. it does so by using pauli matrices as operators and the provided circuits as states. it does the projective measurement part on its own.

# In[5]:


## == A Cost Function (that's to be minimized within the classical optimization circuit)== ##

## == the XX + YY term == ##
def generate_ke_list(n, h):
    ke_gate_list = []
    for i in range(n-1):
        xx_gate = ("XX", [i, i+1], -0.5*h)
        yy_gate = ("YY", [i, i+1], -0.5*h)
        ke_gate_list.extend([xx_gate, yy_gate])
    return ke_gate_list

ke_gate_list = generate_ke_list(n, h)



## == the -Z/2 term == ##
def generate_pe_list(n, h):
    pe_gate_list = []
    for i in range(n):
        Z_gate = ("Z", [i], -0.5*l*(i-(n-1)/2)**2)
        pe_gate_list.extend([Z_gate])
    return pe_gate_list

pe_gate_list = generate_pe_list(n, h)


ham_gates = ke_gate_list + pe_gate_list
###=== Hamiltonian ===###

Ham = SparsePauliOp.from_sparse_list(ham_gates, n)


# Adding the offset due to 2h II (kinetic) and -l*x^2 II/2 (potential)
id_term = sum(l*(i - ((n-1)/2))**2/2 for i in range(n))
off_set = id_term + 2*h



### === Okay we're ready to define that cost function now == ##
def Estimator_Expected(ansatz, n):
    expectation_estim = 0.0
    estimator_expectation = 0.0
    
    amplitudes = map_2n_vec(ansatz)
    norm = np.sqrt(np.dot(amplitudes, amplitudes))
    ansatz = amplitudes/norm
        
    # Define quantum state
    state = QuantumCircuit(n)
    state.initialize(ansatz)
    
    #compute the expectation value <ansatz|Ham|ansatz> without the off_set
    expectation_estim = estimator.run([state], [Ham], shots=32768).result().values
    estimator_expectation = expectation_estim + off_set
    
    return estimator_expectation

