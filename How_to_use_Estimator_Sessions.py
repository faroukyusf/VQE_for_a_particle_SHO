#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from math import pi
from qiskit import *
from qiskit import QuantumCircuit,execute,Aer,IBMQ
from qiskit.compiler import transpile,assemble
from matplotlib import style
from qiskit.circuit.library import QFT
from qiskit.tools.monitor import job_monitor
from itertools import repeat
from qiskit import IBMQ, Aer, transpile, assemble,execute, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.compiler import transpile,assemble
import random
from numpy.random import seed
from numpy.random import randint
import numpy.linalg as linalg
import operator
import time 
import math
import copy
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm
np.set_printoptions(precision=4,suppress=True)


# In[2]:


from qiskit.primitives import Sampler, Estimator

from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import QiskitRuntimeService


# In[3]:


from qiskit_ibm_runtime import Estimator, Session

service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibmq_qasm_simulator")

estimator = Estimator(backend=backend)


# In[4]:


## == Calculate expectation values in an algorithm <psi|Z|psi> == ##

# Step 1: Define operator
c0 = -0.5
op = SparsePauliOp.from_list([("Z", c0),])

# Step 2: Define quantum state
state = QuantumCircuit(1)

state.initialize([1, 0])
# state.x(0)


# In[5]:


##== use Estimator ==##
estimator = Estimator(backend=backend)
job = estimator.run(state, op).result().values 
#Actually, it cn calculate many expectations at once!


# In[6]:


print("estimator expectation: ", job)  #it must be -0.5 since the eigen value of psi=[1,0] is +1


# In[8]:


#  ================= Defining some global variables=======================
lat = 4/197.33  

m_lat = 940*lat   # particle mass

h = 0.5/m_lat     # hopping parameter

w = 8*lat       #omega

l = 0.5*m_lat*w**2   #potential strength 

n = 3


# In[11]:


## == Calculate expectation values in an algorithm <psi|XX+YY|psi> == ##
XX = SparsePauliOp.from_sparse_list([("XX",[0,1], -0.5*h), ("XX",[1,2], -0.5*h)], 3)

YY = SparsePauliOp.from_sparse_list([("YY",[0,1], -0.5*h), ("YY",[1,2], -0.5*h)], 3)

KE = XX + YY


state = QuantumCircuit(n)
psi1 = [0, 0.5, 1/np.sqrt(2), 0, 0.5, 0, 0, 0]
state.initialize(psi1)

job = estimator.run([state], [KE], shots=4096).result().values 


# In[12]:


print("estimator expectations", job) #without the +2h 


# In[14]:


## ====== KE Term = Operator ===== ##
dia = np.eye(n)*-2  #the -2 on the diagonal
arr = randint(0, 1, n-1)+1  
up = np.diag(arr, k=1)   # the 1 on the 1st iagonal
down = np.diag(arr, k=-1)  #the 1 on the -1st diagonal

ke = -h*(dia + up + down)

##### 
eigval, eigvec = linalg.eig(ke)
idx = np.argsort(eigval)
eigval = eigval[idx]    
eigvec = eigvec[:,idx]   

#####

ps1 = eigvec[:,0]
norm0 = np.sqrt(np.dot(ps1, ps1))
ps1 = ps1/norm0

print(ps1) #the exact ground state
print(eigval[0]) #the exact ground state energy


# In[16]:


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

state = map_2n_vec(ps1)


# In[ ]:


from qiskit_ibm_runtime import Options 
 
observable = XX + YY
circuit = QuantumCircuit(n)
circuit.initialize(state)


options = Options() 
options.optimization_level = 1 
options.resilience_level = 1 


service = QiskitRuntimeService() 


# Run on a simulator
backend = service.get_backend("ibmq_qasm_simulator")
# Or use the next line if you want to run on a system
backend = service.least_busy(simulator=False)


with Session(service=service, backend=backend) as session: 
    estimator = Estimator(session=session, options=options) 
    job = estimator.run(circuit, observable) 
    result = job.result() 
    session.close()


# In[19]:


# job results 
print(f" > Expectation value: {result.values}") 


# In[ ]:




