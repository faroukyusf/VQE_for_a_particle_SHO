#!/usr/bin/env python
# coding: utf-8

# In[3]:


# function that takes any n cpomonent vector and map it to 2^n component vector 
def map_2n_vec(state):
    psi = [0] * (2**len(state))
    num = '0' * (len(state)-1)
    for i in range(len(state)):
        nthcomp = state[i]
        binum = num[:i] + "1" + num[i:]
        m = int(binum, 2)
        psi[m] = nthcomp
    return psi


# Here is our cost function. 
# Using the relation method, which is quite robust for certain class of problems including the problem in hand.

# In[5]:


# # ### ============ The classical optimization part using the relaxation method ============== ###

# Number of iterations for the optimization
max_iterations = 35

def classical_optimization_relaxation(ansatz, max_iterations, n):
    psi_temp = copy.deepcopy(ansatz)     #we need a copy the ansatz
    E_est = Estimator_Expected(psi_temp, n)

    psi_updated = np.zeros(n)
    
    for j in range(max_iterations):
        ri = -(n-1)/2    #start point of potential (far left -1)
        rf = (n-1)/2     #end point
        
    #Update each parameter in ansatz individually using the relaxation method
    #first take care of the edges
        start = psi_temp[1]    
        end = psi_temp[n-2]
        psi_updated[0] = (start/2)/(1 + 0.5*(l*ri**2 - E_est)/h)
        psi_updated[n-1] = (end/2)/(1 + 0.5*(l*rf**2 - E_est)/h)
        
     #now take care of every other point   
        for i in range(1, n-1):
            r = i-((n-1)/2)
            V = l*r**2
            site_before = psi_temp[i-1]
            site_after = psi_temp[i+1]
            neighbor_avarage = (site_before + site_after)/2
            psi_updated[i] = neighbor_avarage/(1 + 0.5*(V - E_est)/h)
        
        
        norm = np.sqrt(np.dot(psi_updated, psi_updated))
        psi_updated = psi_updated/norm
        
        E_est = Estimator_Expected(psi_updated, n)
    
    # Retrieve the optimized parameters and energy
    return psi_updated, E_est


# Estimator_Expected is a function that does the quatum routine part using the estimator class. 
# it does so by using pauli matrices as operators and the provided circuits as states. 
# it does the projective measurement part on its own.

# In[ ]:




