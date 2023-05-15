import typing
from typing import Union
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, SamplingVQE, QAOA
# from qiskit.primitives import Sampler
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit.utils import algorithm_globals
# from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session, Sampler
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
algorithm_globals.random_seed = 123456
# from qiskit.algorithms.minimum_eigensolvers import QAOA
# from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit.quantum_info import Pauli, Statevector
from qiskit.result import QuasiDistribution
from qiskit.utils import algorithm_globals
import random

def callback(i, params, obj, stddev, alpha):
    # we translate the objective from the internal Ising representation
    # to the original optimization problem
    objectives[alpha].append(np.real_if_close(-(obj + offset)))

def bitfield(n: int, L: int) -> list[int]:
    result = np.binary_repr(n, L)
    return [int(digit) for digit in result]  # [2:] to chop off the "0b" part

def sample_most_likely(state_vector: Union[QuasiDistribution, Statevector]) -> np.ndarray:
    """Compute the most likely binary string from state vector.
    Args:
        state_vector: State vector or quasi-distribution.

    Returns:
        Binary string as an array of ints.
    """
    if isinstance(state_vector, QuasiDistribution):
        values = list(state_vector.values())
    else:
        values = state_vector
    n = int(np.log2(len(values)))
    k = np.argmax(np.abs(values))
    x = bitfield(k, n)
    x.reverse()
    return np.asarray(x)


def partition(lst, n): 
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def data_process(df):
    ts_na=df.isna().sum()

    wd2=df.drop(columns=list(ts_na[ts_na!=0].index))

    wd2 -= wd2.min()  # equivalent to df = df - df.min()
    wd2 /= wd2.max()

    return wd2

def qaoa_run(mu,sigma,n:int,q:float,budget:int,penalty):
    
    """
    n: number of assets
    q: risk factor
    budget: budget
    penalty: 2 * n  # scaling of penalty term
    
    """
    mdl = Model("portfolio_optimization")
    x = mdl.binary_var_list(range(n), name="x")
    objective = mdl.sum([mu[i] * x[i] for i in range(n)])
    objective -= q * mdl.sum([sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n)])
    mdl.maximize(objective)
    mdl.add_constraint(mdl.sum(x[i] for i in range(n)) == budget)
    qp = from_docplex_mp(mdl)
    
    linear2penalty = LinearEqualityToPenalty(penalty=penalty)
    qp = linear2penalty.convert(qp)
    _, offset = qp.to_ising()
    
    maxiter = 100
    optimizer = COBYLA(maxiter=maxiter)

#     ansatz = RealAmplitudes(n, reps=1)
#     m = ansatz.num_parameters

    sampler = Sampler() #currently using a simulator
    # we might use a noisy simulator eventually and also use qiskit runtime later

    alphas = [0.75]  # confidence levels to be evaluated
    # dictionaries to store optimization progress and results
    objectives = {alpha: [] for alpha in alphas}  # set of tested objective functions w.r.t. alpha
    results = {}  # results of minimum eigensolver w.r.t alpha

    for alpha in alphas:

        # initialize SamplingVQE using CVaR
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            aggregation=alpha,
            callback=lambda i, params, obj, stddev: callback(i, params, obj, stddev, alpha),
        )

        qubit_op=_.primitive
        result = qaoa.compute_minimum_eigenvalue(qubit_op)

        x=bitfield(result.best_measurement['state'],n)


        return x

def qaoa_in_qaoa(df,nqubits):
    n=len(df.columns)
    c=0
    part_num=[]
    while(n>nqubits):
        part_num.append(n)
        n=int(n/nqubits)+1   
        c+=1
    part_num.append(n)
    
    main_li=list(range(len(df.columns)))
    random.shuffle(main_li)
    partitions_ind=[]
    for i in range(len(part_num)-1):
        partitions_ind.append(partition(list(range(part_num[i])),part_num[i+1]))
         
    part_df=[]
    part_df.append(df)
    for j in range(len(partitions_ind)):
        x=pd.concat([part_df[j].iloc[:,partitions_ind[j][i]].mean(axis=1) for i in range(len(partitions_ind[j]))], axis=1)
        part_df.append(x)

    res_sub_qaoa=[]
    for i in range(len(part_df)-1):
        res = []
        for j in partitions_ind[i]:
            mu = np.array(part_df[i].iloc[:,j].mean().values)
            sigma = np.array(part_df[i].iloc[:,j].cov().values)
            x= qaoa_run(mu=mu,sigma=sigma,n=mu.shape[0],q=0.5,budget=3,penalty=2*mu.shape[0])
            res.append(list(x))
        res_sub_qaoa.append(res)
    mu_final=np.array(part_df[-1].mean().values)
    sig_final=np.array(part_df[-1].cov().values)
    x=qaoa_run(mu=mu_final,sigma=sig_final,n=mu_final.shape[0],q=0.5,budget=3,penalty=2*mu_final.shape[0])
    res_sub_qaoa.append(list(x))
    
#     res_tr=res_sub_qaoa
    for i in range(1,len(res_sub_qaoa))[::-1]:
        for j in range(len(res_sub_qaoa[i])):
            for k in range(len(res_sub_qaoa[i][j])):
                if res_sub_qaoa[i][j][k]==0:
                    ind_zero=partitions_ind[i][j][k]
#                     print(ind_zero)
#                     print(partitions_ind[i-1][ind_zero])
#                     print(res_sub_qaoa[i-1][ind_zero])
                    z=res_sub_qaoa[i-1][ind_zero]
                    z1=[abs(x-1) for x in z]
#                     print(z1)
                    res_sub_qaoa[i-1][ind_zero]=z1   
                    
    fin_res = [item for sublist in res_sub_qaoa[0] for item in sublist] 
    fin_ind = [item for sublist in partitions_ind[0] for item in sublist]
    ind_1 = [i for i, j in enumerate(fin_res) if j == 1]
    main_ind1= [fin_ind[i] for i in ind_1]
    
    return list(wd2.columns[main_ind1])