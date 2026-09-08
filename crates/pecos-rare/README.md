# pecos-rare
This crate will be built to perform rare event sampling, targeted primarily at determining
logical error rates for high distance codes and low physical errors. 

# Technical Details
The rare event sampling methods implemented in this library rely on the splitting method,
which will be described here. 

Given an error correcting code (ECC), an error model, and a decoder,
we are interested in computing a logical
error rate $\bar{p}$ for some operation.
We will store the error model as a vector of underlying error probabilities $\vec{p}$.
One can determine $\bar{p}$ by sampling from the error model, passing
the results through the decoder, and determining whether a logical error
has occurred.
A sample can be defined by its unique error event $E\in\Omega$, where $\Omega$
is the set of all possible error events. Further, $F$ is a subset of $\Omega$
containing all events that result in a logical failure.
The probability of an individual
error event $\pi_i(E)$ is defined by the underlying error model $\vec{p}_i$.
From this, the probability of a failure is:
$$\pi_i(F)=\sum_{E\in F} \pi_i(E).$$
As the model's error probabilities shrink and/or the code
distance grows, $\bar{p}$ decreases and estimating it via standard sampling requires an
inversely proportionate number of samples.

## Logical Failure Rate Ratios
To address this cost, we can consider a series of error models $\{\vec{p}_i\}_{i=1}^t$,
where for our use case, we will start with large error rates $\vec{p}_1$ 
and sweep towards a target error rate $\vec{p}_t$. 
This series corresponds to a series of logical error rates $\{\bar{p}_i\}_{i=1}^t$,
where one would expect $\bar{p}_1$ to be large and $\bar{p}_t$ to be the targeted
logical error rate. 
We can compute the $\bar{p}_t$ by using ratios of neighboring logical error rates:
$$\bar{p}_t=\bar{p}_1\prod_{i=1}^{t-1}\frac{\bar{p}_{i+1}}{\bar{p}_{i}}.$$
Note that we will have:
$$\frac{\bar{p}_{i+1}}{\bar{p}_{i}} = \frac{\pi_{i+1}(F)}{\pi_{i}(F)}.$$
We can employ a simple trick to estimate this ratio using a function with 
the property $g(x) = x^{-1}g(x^{-1})$ and a positive constant $C>0$.
Setting $x=C\frac{\pi_i(F)}{\pi_{i+1}(F)}$, we get:
$$\frac{\pi_i(F)}{\pi_{i+1}(F)} = C \frac{\mathbb{E}_{i|F}\left[g(C\frac{\pi_i(F)}{\pi_{i+1}(F)})\right]}{\mathbb{E}_{i+1|F}\left[g(C^{-1}\frac{\pi_{i+1}(F)}{\pi_{i}(F)})\right]}$$
where the expectation values in the numerator/denominator on the right hand side
are numerically computed as:
$$\mathbb{E}_{i|F}\left[g(C\frac{\pi_i(F)}{\pi_{i+1}(F)})\right]=\frac{1}{N}\sum_{\alpha=1}^N g(C^{\pm 1}\frac{\pi_j{E_\alpha}}{\pi_{j\pm 1}(E_\alpha)}).$$
for some number of samples $N$.
By finding $C$ such that:
$$\mathbb{E}_{i|F}\left[g(C\frac{\pi_i(F)}{\pi_{i+1}(F)})\right]=\mathbb{E}_{i+1|F}\left[g(C^{-1}\frac{\pi_{i+1}(F)}{\pi_{i}(F)})\right]$$
we get $\frac{\pi_i(F)}{\pi_{i+1}(F)} = C$, having successfully computed the ratio of failure rates.

## Metropolis Algorithm
Now, the remaining challenge is to correctly sample logical failure events $E_\alpha$.
We will use a Metropolis algorithm to perform Markov chain Monte Carlo
random walk between events $E_\alpha\in F$.
To ensure the correct distribution is sampled, we requrie that the sampling
technique is irreducible and that it satisfies detailed balance, i.e.
$$\frac{\text{Pr}(E')}{\text{Pr}(E)}=\frac{\text{Pr}(E'|E)}{\text{Pr}(E|E')}.$$
This ensures that at steady state the probability flow from $\text{Pr}(E\to E')=\text{Pr}(E)\text{Pr}(E'|E)$ is
equal to the flow from $\text{Pr}(E'\to E)=\text{Pr}(E')\text{Pr}(E|E')$. 

### Monte Carlo Rules
In the existing literature, separate Metropolis flip and acceptance rules 
have been proposed depending on the error model. I think
that to visit all $E\in F$ one may require more complex flip
and acceptance rules depending on the code and error models.
Here I'll cover the basics of the flip rules and acceptance rates
that are being implemented.

#### Circuit Error Model Flip Rules
The set of errors are now defined by vectors of gate and fault pairs, 
$E=(\vec{g}, \vec{f})$, which specify which
gate $\vec{g}$ are experiencing corresponding faults $\vec{f}$.
The algorithm proceeds by:
* Initialize with a failure event $E\in F$. 
* Until converged:
  * Select a gate-fault pair $(g_\alpha, f_\alpha)$
  * Define the new event $E'$:
    * If $g_\alpha\notin \vec{g}$ then $E'=E\cup (g_\alpha, f_\alpha)$
    * Otherwise, it means there is some $(g_\alpha, h)\in E$. In this case, we want to either remove the event if $h=f_\alpha$ or change the event ($(g_\alpha, h)\to(g_\alpha, f_\alpha)$), i.e. $E'=(E\cup (g_\alpha, f_\alpha)) \backslash (g_\alpha, h)$.
  * Accept or Reject the flip:
    * If $E'\notin F\to$ reject the flip. 
    * Else if $g_\alpha\notin \vec{g}$:
      * $q=\frac{\text{Pr}(g)}{1-\text{Pr}(g)}\text{Pr}_g(f)$
      * where $\text{Pr}(g)$ is the probability of gate $g$ failing and $\text{Pr}_g(f)$ is the probability of fault $f$ given that $g$ has failed. 
    * Else if $h=f_\alpha\to q=1$
    * Else $q=\text{Pr}_g(f)$.

A proof that these flip rules satisfy local detailed balance is in
[Rare Event Simulation of Quantum Error-Correcting Circuits](https://arxiv.org/pdf/2509.13678).

[#### Decoder Graph Edges
Consider a QECC defined by a check matrix $H\in\mathbb{F}_2^{M\times N}$
with a decoder $D: \mathbb{F}_2^M\to\mathbb{F}_2^N$. 
The rows of $H$ correspond to independent errors with probability $q$,
which then flip the checks in the corresponding column of $H$. 
An event $E$, as discussed earlier, is specified $E\in\mathbb{F}_2^N$
and produces the syndrome $\sigma=HE$. 
The decoder provides a correction $d=D(\sigma)$ such that
$Hd=\sigma$. If $Ad=Ae$, then t
When a set of errors $e\in \mathbb{F}_2^N$ occurs]: #

# References
Details of this algorithm and related approaches can be found in the following papers:
- [Simulation of rare events in quantum error correction](https://arxiv.org/pdf/1308.6270)
- [Rare Event Simulation of Quantum Error-Correcting Circuits](https://arxiv.org/pdf/2509.13678)
- [Fail fast: techniques to probe rare events in quantum error correction](https://arxiv.org/pdf/2511.15177)

# Implementation Plan:
- [ ] Implement fundamental structures for circuit-level fault models:
  - [x] Figure out how to provide default labels to all of the gates in a defined circuit
  - [x] Individual gate identifiers
    - [x]  Need to define $\text{Pr}(g)$
  - [x] Figure out all the places where errors can be added to gates in a defined circuis
  - [x]  Individual fault identifiers
    - [x]  Need to define $\text{Pr}_g(f)$
  - [x]  Individual gate-fault pairs
    - [x]  Need to define $\text{Pr}(g)$ and $\text{Pr}_g(f)$
  - [x]  Events (collections of gate-fault pairs)
    - [ ]  Method to check if $E\in F$
- [ ]  Implement Metropolis core
  - [ ]  Allow for flexible definition of flips
  - [ ]  Allow for flexible definition of acceptance criteria
  - [ ]  Write customized version for circuit-level models
- [ ]  Implement Splitting core
  - [ ]  Define probability sequence
  - [ ]  Develop method to compute $C$ (and error on $C$)
  - [ ]  Allow for flexible definition of $g(x)$. 
  - [ ]  Write customized version for circuit-level models
- [ ]  Create a public rust API
- [ ]  Add CLI
- [ ]  Add Python interface
- [ ]  Documentation
- [ ]  Demonstrations
- [ ]  Benchmarking/Validation

# Ideas for future work
- Improve current approach:
- Other literature approaches:
- Explore other rare event simulation techniques:
