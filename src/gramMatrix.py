import numpy as np
from cvxopt import matrix
from cvxopt.solvers import lp


def is_square(x):
    """Returns if x is +/- a square number"""

    sqrt_abs = np.sqrt(abs(x))
    sqrt_abs_rnd = int(np.round(sqrt_abs))

    return sqrt_abs_rnd**2 == abs(x)

def get_kernel_basis(matrix):
    """Returns an integer basis for the kernel of `matrix` using Gaussian elimination with integer arithmetic."""

    if len(matrix) == 0:
        return []

    # Number of rows and columns
    nr = len(matrix)
    nc = len(matrix[0])

    # Augment by an identity matrix
    matrix_aug = np.block([matrix, np.identity(nr, dtype='int')])

    # Perform Gaussian elimination
    pivot_row, pivot_col = 0, 0
    while pivot_row < nr and pivot_col < nc:

        # Check if pivot is zero
        if matrix_aug[pivot_row,pivot_col] == 0:

            # Find nonzero elements below of pivot
            swap_row = pivot_row
            while swap_row < nr and matrix_aug[swap_row,pivot_col] == 0:
                swap_row += 1

            if swap_row < nr:
                # Perform the swap of rows
                matrix_aug[pivot_row], matrix_aug[swap_row] = matrix_aug[swap_row].copy(), matrix_aug[pivot_row].copy()

        if matrix_aug[pivot_row,pivot_col] != 0:
            # Use non-zero pivot for elimination

            sign = np.sign(matrix_aug[pivot_row,pivot_col])

            for jj in range(pivot_row+1, nr):
                gcd = np.gcd(matrix_aug[pivot_row,pivot_col], matrix_aug[jj,pivot_col])
                matrix_aug[jj] = sign * (matrix_aug[pivot_row,pivot_col]*matrix_aug[jj] - matrix_aug[jj,pivot_col]*matrix_aug[pivot_row]) / gcd

                # Reduce row by gcd
                gcd = np.gcd.reduce(matrix_aug[jj])
                matrix_aug[jj] = matrix_aug[jj] / gcd

            pivot_row += 1
        
        pivot_col += 1

    nullity = np.count_nonzero(np.max(np.abs(matrix_aug[:,:nc]), axis=1) == 0)
    
    if nullity == 0:
        kernel_basis = []
    else:
        kernel_basis = matrix_aug[(-nullity):, nc:]

    return kernel_basis

def signature(matrix):
    """Returns the signature, (n_+, n_-, n_0), of `matrix`."""

    if len(matrix) == 0 or len(matrix[0]) == 0:
        # Empty matrix (0x0)
        return 0, 0, 0

    # Get eigenvalues, rank and nullity
    eigs = np.linalg.eigvalsh(matrix)
    rank = np.linalg.matrix_rank(matrix)
    nullity = len(matrix) - rank

    # Correct fp errors by setting (near-)zero eigenvalues
    # to exactly zero based on the known nullity
    order = np.argsort(abs(eigs))
    eigs = eigs[order]
    eigs[:nullity] = 0

    # Count number of positive and negative eigenvalues
    n_pos = np.count_nonzero(eigs > 0)
    n_neg = np.count_nonzero(eigs < 0)

    return n_pos, n_neg, nullity

def requires_npos_zero(gram_b0bi, gram_bibj):
    """Returns whether a convex linear combination of the vectors bi is null and orthogonal to all bI=(b0,bi)."""

    # First check to see if conv(bi) contains a null vector orthogonal
    # to all of bI=(b0,bi). Such a vector corresponds to an element of
    # the kernel of the k×(k+1) matrix
    #     gram_aug = [b0.bi, bi.bj]
    # with all non-negative components. If such a vector exists then
    # we must have n_pos(G) = 0, since otherwise this null vector is
    # actually the zero vector and would imply a violation of j.bi > 0.
    # In particular, this requries T ≥ 9.
    gram_aug = np.block([[gram_b0bi], [gram_bibj]]).T
    kernel_basis = get_kernel_basis(gram_aug)

    if len(kernel_basis) == 0:
        null_conv_bi_exists = False

    else:
        id = np.identity(len(kernel_basis))

        # The kernel basis of gram_aug is a collection of vectors x_a with components x_a^i
        # which give null vectors u_a = s_a^i b_i orthogonal to all bI.

        # Look for a vector in the kernel, x = x^a s_a, with all components non-negative,
        # i.e. x^a s_a^i ≥ 0 for each i (and not all x^a zero).

        # Minimize -sum(x^a s_a^i, i) ...
        c = -np.sum(kernel_basis, axis=1)

        # ...subject to x^a ∈ [-1,1] and x^a s_a^i ≥ 0 for each i.
        G = np.block([[id], [-id], [-kernel_basis.T]])
        h = [*np.ones(2*len(id)), *np.zeros(len(kernel_basis[0]))]

        c = matrix(c, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        
        # Optimize!
        soln = lp(c, G, h, options={'show_progress': False})
        x_opt = soln['x']

        # If the optimal solution is at (x_opt)^a = 0 then there is no
        # positive linear combination of the bi which is null and orthogonal to all bI.
        # If there *is* such a linear combination, then the solution will be at the boundary,
        # i.e. max[(x_opt)^a] = ±1: use 0.5 as the decision boundary.
        null_conv_bi_exists = max(abs(x_opt)) > 0.5

    return null_conv_bi_exists

def get_T_min(gram_b0bi, gram_bibj, T_min_start=0, T_min_max=np.inf):
    """Returns the smallest value of T such that the Gram matrix built from b0bi and bibj is admissible.

    The Gram matrix
    G = [[9-T   b0.bi]
        [b0.bi  bi.bj]]
    must have, at a minimum, n(+) ≤ 1 and n(-) ≤ T. If there is a convex linear combination of bi
    which is null and orthogonal to all bI=(b0,bi), then we need n(+) = 0 and n(-) < T.
    Other than the necessary condition that |det(G)| be a square number when n(+) = 1 and n(-) = T,
    details of the unimodular lattice embedding are not checked in order to keep the computation fast.
    Requiring that the lattice corresponding to G has an embedding into a unimodular lattice increases T
    by at most two.

    Args:
        gram_b0bi (list): 1d array of b0.bi inner products.
        gram_bibj (list): Symmetric integer matrix of bi.bj inner products.
        T_min_start (int, optional): Value of T at which to being checking for admissibility. Defaults to 0.
        T_min_max (int, optional): Maximum allowed value for T_min (will abort if surpassed). Defaults to np.inf.

    Returns:
        int: T_min
    """

    if len(gram_b0bi) == 1:
        if gram_bibj[0][0] == 0 and gram_b0bi[0] == 0:
            return 9

    needs_npos_zero = requires_npos_zero(gram_b0bi, gram_bibj)

    T_min = T_min_start

    if needs_npos_zero:
        # There cannot be a positive eigenvalue, else j.bi > 0 is violated
        # In particular, b0.b0 = 9 - T ≤ 0  ⇒  T ≥ 9
        T_min = max(T_min, 9)

    gram_bIbJ = np.pad(gram_bibj, (1,0))
    gram_bIbJ[0,1:] = gram_b0bi
    gram_bIbJ[1:,0] = gram_b0bi

    gram_bIbJ[0,0] = 9 - T_min

    # Remove bi until no more linear dependencies in bi.bJ
    kernel_basis = get_kernel_basis(gram_bIbJ[1:])

    while len(kernel_basis) > 0:
        ii = np.where(kernel_basis[0])[0][0]
        gram_bIbJ = np.delete(gram_bIbJ, ii+1, 0)
        gram_bIbJ = np.delete(gram_bIbJ, ii+1, 1)
        kernel_basis = get_kernel_basis(gram_bIbJ[1:])

    # Get signature
    n_pos_bIbJ, n_neg_bIbJ, n_0_bIbJ = signature(gram_bIbJ)

    # If det(bi.bj) ≠ 0 then det(bI.bJ) = (9-T)*det(bi.bj) - v*adj(bi.bj)*v
    # will have opposite sign of det(bi.bj) for large enough T, ensuring
    # that n_neg(bI.bJ) = n_neg(bi.bj) + 1 and thus n_pos(bI.bJ) = n_pos(bi.bj) ≤ 1
    # Otherwise, if det(bi.bj) = 0 then det(bI.bJ) is independent of T. Since we
    # have removed all linear dependencies in bi.bJ, det(bI.bJ) ≠ 0 and the number
    # of positive eigenvalues n_pos(bI.bJ) is constant
    rank_bibj = np.linalg.matrix_rank(gram_bIbJ[1:,:][:,1:])

    if rank_bibj < len(gram_bIbJ[1:]) and (n_pos_bIbJ > 1 or (n_pos_bIbJ == 1 and needs_npos_zero)):
        # In these cases, n_pos(bI.bJ) > 1 for all T
        return +np.inf
    
    # If n_pos_bIbJ is currently too large, jump ahead to the point where det(G) crosses zero.
    if n_pos_bIbJ > 1 or (n_pos_bIbJ == 1 and needs_npos_zero):
        # Write det(G) = (9-T)det(g) + const
        # Here det(g) != 0, so T >= 9 + const/det(g)
        detG = int(np.round(np.linalg.det(gram_bIbJ)))
        detg = int(np.round(np.linalg.det(gram_bIbJ[1:,:][:,1:])))
        const = detG - (9-T_min)*detg
        T_min = max(T_min, 9 + (const*detg) // (detg**2))
        gram_bIbJ[0,0] = 9 - T_min
        n_pos_bIbJ, n_neg_bIbJ, n_0_bIbJ = signature(gram_bIbJ)

    det = int(np.round(np.linalg.det(gram_bIbJ)))

    # Increment T until the eigenvalue bounds are satisfied
    while T_min <= T_min_max \
        and (n_pos_bIbJ > 1 or n_neg_bIbJ > T_min \
            or (n_neg_bIbJ == T_min and n_pos_bIbJ == 1 and not is_square(det)) \
            or (n_pos_bIbJ == 1 and needs_npos_zero)):
        T_min += 1
        gram_bIbJ[0,0] = 9 - T_min
        n_pos_bIbJ, n_neg_bIbJ, n_0_bIbJ = signature(gram_bIbJ)
        det = int(np.round(np.linalg.det(gram_bIbJ)))

    return T_min
