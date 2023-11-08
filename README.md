# 6d-sugra-classify

Classification of anomaly-free 6D supergravity theories.

A multigraph can be constructed for some choice of simple groups and upper bounds on $\Delta=H_\text{ch}-V$. Vertices represent simple admissible theores and edges represent mergings of the hypermultiplets of the two incident vertices which result in an admissible $k=2$ theory. Admissible theories with $k$ simple factors then correspond to $k$-cliques in the multigraph. Irreducible, admissible $k$-cliques satisfying the bound $\Delta+28n_-^\mathfrak{g} \leq 273$ can be enumerated using a "branch-and-prune" algorithm.

See [[2311.00868](https://arxiv.org/abs/2311.00868)].

Some notes on implementation:

 - With our normalization, A, B and C are almost always integers. The only exceptions are for B3 and D4 where C is not an integer. However, for these groups C + 3B/4 *is* an integer and is stored in place of C everywhere. Sums over C are unaffected provided the B-constraint is satisfied.
 - For quaternionic representations free of Witten anomalies H,A,B,C are always even and are divided by two so that nR for these represents the number of half-hypermultiplets. The irrep ID ends with '_h' to indicate it is a half-hypermultiplet.
 - Some vertices are removed by hand:
   - All type-B vertices with $\Delta_i<-29$ (which occur only for exceptional groups)
   - $\{\mathrm{SU}(2N), (2N+8)\times\underline{\mathbf{2N}} + \underline{\mathbf{2N(2N-1)/2}}\}$ when $\mathrm{Sp}(N)$ vertices are present, since it is essentially identical to $\{\mathrm{Sp}(N), (2N+8)\times\underline{\mathbf{2N}}\}$
   - $\{\mathrm{SU}(2N), 16\times\underline{\mathbf{2N}} + 2\times\underline{\mathbf{2N(2N-1)/2}}\}$ when $\mathrm{Sp}(N)$ vertices are present, since it is essentially identical to $\{\mathrm{Sp}(N), 16\times\underline{\mathbf{2N}} + \underline{\mathbf{(N-1)(2N+1)}}\}$
