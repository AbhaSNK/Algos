## Process for QAOA in QAOA

The process can seem a little complicated mainly because it involves several QAOA models.
Below are the steps and some notes:

### Some notes for this algorithm's implementation:

1. I think for now, instead of using real quantum devices, we could use simulators+noise models (https://qiskit.org/documentation/apidoc/aer_noise.html) or fake backends (https://qiskit.org/documentation/apidoc/providers_fake_provider.html). We have previously used fake backends, but I realized that they are snapshots of the system, which could be from several years ago. So it might be a better idea to use simulator + noise models, which have most recent calibrations. For simulator+noise models, the qiskit fall challenge 2022 is a good resource, we can provide noise model in the Options for Sampler: https://github.com/qiskit-community/ibm-quantum-challenge-fall-22/tree/main/content

2. Things have changed in Qiskit a lot now, most of the things use Qiskit primitives and maybe, using IBM Quantum runtime for QAOA would be our way to go ahead. The tutorial we referred to previously, also uses Qiskit sampler:https://qiskit.org/documentation/optimization/tutorials/08_cvar_optimization.html with VQE, we have to use QAOA similarly (https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/algorithms/minimum_eigensolvers/qaoa.py), but we need to provide a Sampler with an appropriate backend, we can do that like this: https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials/how-to-getting-started-with-sampler.html I have tried the fake backend approach and have seen that sampler works with that, so hoping that we won't face issues going forward. It should also work for simulator+noise model.

### Steps:

1. We would scale each stock closing prices between -1 and 1 through a simple linear approach.
2. Suppose there are 500 stocks and we only can use 5 qubits (we might be able to use more. Maybe, we can try for a paid IBM service later on for IBM Falcon with 27 qubits, but that seems expensive: 1.6 USD/sec). We will randomly divide our portfolio into sub-portfolios with 5 stocks each. This would give us 100 sub-portfolios.
3. We would run QAOA for each of these 100 sub-portfolios. There is a way to run mutiple jobs in a Session, but we would have to figure out how that can be done for the algorithm itself. If we can't, we can run them separately.
4. Next, 100/5=20, which is still greater than 5, so we would again run 20 QAOA models. We would divide 100 into 20 bigger sub-portfolios randomly. This is the 1st stage of merging.
5. Suppose the results of 5 smaller portfolios within one of this bigger portfolio are [1,1,0,0,1], [0,1,1,0,0], [1,0,0,0,0], [0,0,1,0,1] and [1,0,1,0,1]. Now, QAOA has to be run for each of these 20 bigger sub-portfolios by considering each smaller sub-portfolio as 1 single node. The mean for each small sub-portfolio would be taken as mean of means (mean taken for a sub-portfolio each day). The covariance matrix between smaller sub-portfolios would be calculated by taking cov between each time series from 1 with that of the other and averaging it out (average of 5\*5=25 covariances) and the covariance for 1 small sub-portfolio would be the average of upper triangle of the cov matrix within that smaller sub-portfolio.
6. Suppose the QAOA output for 1 bigger sub-portfolio is [1,1,0,1,0], then the interpretation of this would be to take the smaller portfolio result as it is for all 1s, but negating the output for all the 0s.
7. Since 20>5, we would have to again create bigger portfolios with 20/5=4, i.e combining 5 of the bigger sub-portfolios and the final interpretation would be similar to how we concluded in step 6. It would end here, since 4<5.

## steps for Quantum noise-induced reservoir computing

1. We would again be using the scaled time series (scaled between -1 and 1).
2. For each of the 1s in the final output from the previous algorithm, we would want to make predictions for 5 time steps.
3. We would be using an density matrix simulator and use 5 qubits.
4. We need to choose a window, which could be 20. This would mean we would be creating 20 parametric circuits and binding values as dicussed in the Algorithms_approaches.doc. (RX, RZZ and noise channels).
5. We would be calculating expectations w.r.t each qubit by taking a trace of the reduced density matrix w.r.t each qubit.
6. The above would give us a 20 X 5 matrix with scaled target for the 20 time steps.
7. We will perform a linear regression (classical) and get 5 weights, using which we will again create 5 more parametric circuits and perform predictions for 5 time steps.
8. The predictions would have to be rescaled back to the original scale of the time series.

### Overall output as shown to users

We would show the stocks as shown as favorable by our 1st algorithm and then, predictions for those using the 2nd algorithm. We could also show animated or static plots for historical data and predictions.
