import numpy as np
from itertools import combinations
from tqdm import tqdm
import os

from vertex import Vertex
from constraintSolver import ConstraintSolver


# Padding for computing Hbounds
B_PAD = 5

# (d)ata(t)ype for storing all irrep data (quat = 'quaternionic' and hw_vec = 'highest-weight vector')
dt_irrep = np.dtype([('ID', 'str', 255), ('H', 'int'), ('A', 'int'), ('B', 'int'), ('C', 'int'),
                        ('quat', 'bool'), ('hw_vec', 'str', 255)])

# Roots for exceptional groups
ROOTS_E6 = [
    [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1], [0, 1, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],

    [0, 1, 2, 1, 0, 1],
    [1, 1, 2, 1, 0, 1], [0, 1, 2, 1, 1, 1],
    [1, 1, 2, 1, 1, 1],

    [1, 2, 2, 1, 0, 1], [0, 1, 2, 2, 1, 1],
    [1, 2, 2, 1, 1, 1], [1, 1, 2, 2, 1, 1],
    [1, 2, 2, 2, 1, 1],
    [1, 2, 3, 2, 1, 1],
    [1, 2, 3, 2, 1, 2]
]

ROOTS_E7 = [
    [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0],

    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    
    [0, 0, 1, 2, 1, 0, 1],
    [0, 0, 1, 2, 1, 1, 1], [0, 1, 1, 2, 1, 0, 1],
    [0, 1, 1, 2, 1, 1, 1], [1, 1, 1, 2, 1, 0, 1],
    [1, 1, 1, 2, 1, 1, 1],

    [0, 0, 1, 2, 2, 1, 1], [0, 1, 2, 2, 1, 0, 1],
    [0, 1, 1, 2, 2, 1, 1], [0, 1, 2, 2, 1, 1, 1], [1, 1, 2, 2, 1, 0, 1],
    [1, 1, 1, 2, 2, 1, 1], [1, 1, 2, 2, 1, 1, 1],
    
    [0, 1, 2, 2, 2, 1, 1], [1, 2, 2, 2, 1, 0, 1],
    [1, 1, 2, 2, 2, 1, 1], [1, 2, 2, 2, 1, 1, 1],
    
    [1, 2, 2, 2, 2, 1, 1],
    
    [0, 1, 2, 3, 2, 1, 1],
    [1, 1, 2, 3, 2, 1, 1],
    [1, 2, 2, 3, 2, 1, 1],
    
    [0, 1, 2, 3, 2, 1, 2],
    [1, 1, 2, 3, 2, 1, 2],
    [1, 2, 2, 3, 2, 1, 2],
    [1, 2, 3, 3, 2, 1, 1],
    [1, 2, 3, 3, 2, 1, 2],
    [1, 2, 3, 4, 2, 1, 2], 
    [1, 2, 3, 4, 3, 1, 2],
    [1, 2, 3, 4, 3, 2, 2]
]

ROOTS_E8 = [
    [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1], [0, 1, 2, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 1], [1, 1, 2, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1], [0, 1, 2, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 2, 1, 1, 0, 0, 1], [1, 2, 2, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 2, 1, 1, 1, 0, 1], [0, 1, 2, 2, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 2, 1, 1, 1, 0, 1],
    [1, 2, 2, 1, 1, 0, 0, 1], [1, 1, 2, 2, 1, 0, 0, 1], [0, 1, 2, 1, 1, 1, 1, 1], [0, 1, 2, 2, 1, 1, 0, 1],
    [1, 1, 2, 1, 1, 1, 1, 1], [1, 2, 2, 1, 1, 1, 0, 1], [1, 1, 2, 2, 1, 1, 0, 1], [1, 2, 2, 2, 1, 0, 0, 1],
    [0, 1, 2, 2, 1, 1, 1, 1], [0, 1, 2, 2, 2, 1, 0, 1], [1, 2, 2, 1, 1, 1, 1, 1], [1, 1, 2, 2, 1, 1, 1, 1],
    [1, 2, 2, 2, 1, 1, 0, 1], [1, 1, 2, 2, 2, 1, 0, 1], [1, 2, 3, 2, 1, 0, 0, 1], [0, 1, 2, 2, 2, 1, 1, 1],
    [1, 2, 2, 2, 1, 1, 1, 1], [1, 1, 2, 2, 2, 1, 1, 1], [1, 2, 3, 2, 1, 1, 0, 1], [1, 2, 2, 2, 2, 1, 0, 1],
    [1, 2, 3, 2, 1, 0, 0, 2], [0, 1, 2, 2, 2, 2, 1, 1], [1, 2, 3, 2, 1, 1, 1, 1], [1, 2, 2, 2, 2, 1, 1, 1],
    [1, 1, 2, 2, 2, 2, 1, 1], [1, 2, 3, 2, 2, 1, 0, 1], [1, 2, 3, 2, 1, 1, 0, 2], [1, 2, 3, 2, 2, 1, 1, 1],
    [1, 2, 3, 2, 1, 1, 1, 2], [1, 2, 2, 2, 2, 2, 1, 1], [1, 2, 3, 3, 2, 1, 0, 1], [1, 2, 3, 2, 2, 1, 0, 2],
    [1, 2, 3, 3, 2, 1, 1, 1], [1, 2, 3, 2, 2, 2, 1, 1], [1, 2, 3, 2, 2, 1, 1, 2], [1, 2, 3, 3, 2, 1, 0, 2],
    [1, 2, 3, 3, 2, 2, 1, 1], [1, 2, 3, 3, 2, 1, 1, 2], [1, 2, 3, 2, 2, 2, 1, 2], [1, 2, 4, 3, 2, 1, 0, 2],
    [1, 2, 3, 3, 3, 2, 1, 1], [1, 2, 3, 3, 2, 2, 1, 2], [1, 2, 4, 3, 2, 1, 1, 2], [1, 3, 4, 3, 2, 1, 0, 2],
    [1, 2, 3, 3, 3, 2, 1, 2], [1, 2, 4, 3, 2, 2, 1, 2], [1, 3, 4, 3, 2, 1, 1, 2], [2, 3, 4, 3, 2, 1, 0, 2],
    [1, 2, 4, 3, 3, 2, 1, 2], [1, 3, 4, 3, 2, 2, 1, 2], [2, 3, 4, 3, 2, 1, 1, 2], [1, 3, 4, 3, 3, 2, 1, 2],
    [1, 2, 4, 4, 3, 2, 1, 2], [2, 3, 4, 3, 2, 2, 1, 2], [2, 3, 4, 3, 3, 2, 1, 2], [1, 3, 4, 4, 3, 2, 1, 2],
    [2, 3, 4, 4, 3, 2, 1, 2], [1, 3, 5, 4, 3, 2, 1, 2], [2, 3, 5, 4, 3, 2, 1, 2], [1, 3, 5, 4, 3, 2, 1, 3],
    [2, 4, 5, 4, 3, 2, 1, 2], [2, 3, 5, 4, 3, 2, 1, 3], [2, 4, 5, 4, 3, 2, 1, 3], [2, 4, 6, 4, 3, 2, 1, 3],
    [2, 4, 6, 5, 3, 2, 1, 3], [2, 4, 6, 5, 4, 2, 1, 3], [2, 4, 6, 5, 4, 3, 1, 3], [2, 4, 6, 5, 4, 3, 2, 3]
]

ROOTS_F4 = [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
    [1, 1, 1, 0], [0, 1, 1, 1],
    [1, 1, 1, 1],
    
    [0, 1, 2, 0],
    [0, 1, 2, 1], [0, 1, 2, 2], [1, 1, 2, 0],
    [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 2, 0], [1, 2, 2, 1], [1, 2, 2, 2], [1, 2, 3, 1], [1, 2, 3, 2],
    [1, 2, 4, 2], [1, 3, 4, 2], [2, 3, 4, 2]
]

ROOTS_G2 = [[1, 0], [0, 1], [1, 1], [1, 2], [1, 3], [2, 3]]


class SimpleGroup():

    def __init__(self, cartan_type, n):
        """Initializes the simple group of the given Cartan type (A,B,C,D,E,F,G) and rank `n`.

        The classical groups are A[n] ~ SU(n+1), B[n] ~ SO(2n+1), C[n] ~ Sp(n) and D[n] ~ SO(2n).
        There are also five exceptional groups, E[6], E[7], E[8], F[4] and G[2].

        Taking into account the low-rank isomorphisms, the allowed groups are
            A[n ≥ 1] → SU(2) SU(3) SU(4) SU(5) SU(6)  SU(7)  SU(8)  SU(9)  ...
            B[n ≥ 3] →             SO(7) SO(9) SO(11) SO(13) SO(15) SO(17) ...
            C[n ≥ 2] →       Sp(2) Sp(3) Sp(4) Sp(5)  Sp(6)  Sp(7)  Sp(8)  ...
            D[n ≥ 4] →                   SO(8) SO(10) SO(12) SO(14) SO(16) ...
            E,F,G    →       G[2]        F[4]         E[6]   E[7]   E[8] 

        Args:
            cartan_type (char): Family in the Killing-Cartan classification.
            n (int): Lie group rank.

        Raises:
            Exception: Invalid group.

        """

        # Check that the group is allowed
        valid =    (cartan_type == 'A' and n >= 1        ) \
                or (cartan_type == 'B' and n >= 3        ) \
                or (cartan_type == 'C' and n >= 2        ) \
                or (cartan_type == 'D' and n >= 4        ) \
                or (cartan_type == 'E' and n in [6, 7, 8]) \
                or (cartan_type == 'F' and n == 4        ) \
                or (cartan_type == 'G' and n == 2        )
        
        if not valid:
            err = f'Invalid group {cartan_type}[{n}].'
            err += '\nThe allowed groups are '
            err += 'A[n≥1], B[n≥3], C[n≥2], D[n≥4], E[6,7,8], F[4] and G[2].'
            raise Exception(err)
        

        # Placeholder for irreps
        self.irreps = None

        # Start arrays of Amax, Amax_hat values
        # These will be populated with values relevant for bounding bi.bj
        # and determining which type-A vertices can possibly have nontrivial edges
        self.AmaxG     = [0]
        self.AmaxG_hat = [0]

        # Group information
        self.cartan_type = cartan_type
        self.n = int(n)
        self.ID = self.cartan_type + str(self.n).rjust(2, '0')

        # Finally, set the following data:
        #  - The dimension, i.e. number of vectors
        #  - Trace indicies A,B,C for adjoint
        #  - Index data used later when computing the trace indicies A,B,C for all irreps
        #  - A boolean recording whether quaternionic irreps exist

        if self.cartan_type == 'A':

            # A[n] ~ SU(n+1)
            self.V = (self.n+1)**2 - 1
            
            # A,B,C indices for adjoint
            self.A_adj = 2*(self.n + 1)
            self.B_adj = 2*(self.n + 1)
            self.C_adj = 6

            # Used in calculating A,B,C indices for all irreps
            self.J4_adj = (self.n+1) * ((self.n+1)**2 - 4) * ((self.n+1)**2 - 9) / 3
            self.CmKA2_adj_numerator = 6*(self.V + 2) - 10 * (self.n+1)**2
            self.CmKA2_adj_denominator = (self.V + 2)
            
            # Quaternionic irreps only for A5~SU(6), A9~SU(10), A13~SU(14), ...
            self.has_quaternionic = (self.n % 4 == 1)

            if self.n in [1, 2]:
                self.B_adj = 0
                self.C_adj = 2*(self.n+7)  # 'C' = 2C = integer

        elif self.cartan_type == 'B':

            # B[n] ~ SO(2n+1)
            self.V = self.n*(2*self.n+1)

            # A,B,C indices for adjoint
            self.A_adj = 2*(2*self.n - 1)
            self.B_adj = 4*(2*self.n - 7)
            self.C_adj = 12

            # For B2~SO(5) and B3~SO(7), C is replaced everywhere by C+3B/4
            if self.n in [2, 3]:
                self.C_adj = 6*self.n - 9

            # Used in calculating A,B,C indices for all irreps
            self.J4_adj = (self.n**2 - 1)*(2*self.n - 1)*(2*self.n + 3)*(2*self.n - 7) / 12
            self.CmKA2_adj_numerator = 12*(self.V + 2) - 10 * (2*self.n - 1)**2
            self.CmKA2_adj_denominator = (self.V + 2)

            # Quaternionic irreps only for B2~SO(5), B5~SO(11), B6~SO(13), B9~SO(19), ...
            self.has_quaternionic = (self.n % 4 in [1, 2])

        elif self.cartan_type == 'C':

            # C[n] ~ Sp(n)
            self.V = self.n*(2*self.n+1)
            self.A_adj = 2*(self.n + 1)
            self.B_adj = 2*(self.n + 4)
            self.C_adj = 3
            
            # Used in calculating A,B,C indices for all irreps
            self.J4_adj = (self.n**2 - 1)*(2*self.n - 1)*(2*self.n + 3)*(self.n + 4) / 6
            self.CmKA2_adj_numerator = 3*(self.V + 2) - 10 * (self.n + 1)**2
            self.CmKA2_adj_denominator = (self.V + 2)

            # Quaternionic irreps for all Cn
            self.has_quaternionic = True

        elif self.cartan_type == 'D':
            
            # D[n] ~ SO(2n)
            self.V = self.n*(2*self.n-1)
            self.A_adj = 4*(self.n - 1)
            self.B_adj = 8*(self.n - 4)
            self.C_adj = 12

            # Used in calculating A,B,C indices for all irreps
            self.J4_adj = (self.n**2 - 1)*(2*self.n + 1)*(2*self.n - 3)*(self.n - 4) / 6
            self.CmKA2_adj_numerator = 12*(self.V + 2) - 40 * (self.n - 1)**2
            self.CmKA2_adj_denominator = (self.V + 2)

            # Quaternionic irreps only for D6~SO(12), D10~SO(20), D14~SO(28), ...
            self.has_quaternionic = (self.n % 4 == 2)

        elif self.cartan_type == 'E':
            # E[6], E[7] and E[8]

            # These quadratic polynomials (Lagrange) interpolate the three values
            self.V = 1008 - 335*self.n + 30*self.n**2   # V(6) = 78, V(7) = 133, V(8) = 248

            self.A_adj = 6*(34 - 11*self.n + self.n**2) # A(6) = 24, A(7) = 36, A(8) = 60
            self.B_adj = 0
            self.C_adj = 3*(36 - 11*self.n + self.n**2) # C(6) = 18, C(7) = 24, C(8) = 36

            self.has_quaternionic = (self.n == 7)

        elif self.cartan_type == 'F':
            # F[4]
            self.V = 52

            self.A_adj = 18
            self.B_adj = 0
            self.C_adj = 15

            self.has_quaternionic = False

        elif self.cartan_type == 'G':
            # G[2]
            self.V = 14

            self.A_adj = 8
            self.B_adj = 0
            self.C_adj = 10

            self.has_quaternionic = False

    def display_irreps(self, H_min=0, H_max=np.inf):
        """Display data for irreps with H in the range [`H_min`, `H_max`]."""

        # Construct the table header
        print('{:15}{:>6}{:>8}{:>8}'.format('ID', 'H', 'A', 'B'), end='')
        
        if self.ID in ['B03', 'D04']:    # For these groups C is actually C+3B/4 ∈ Z
            print('{:>8}'.format('C+3B/4'), end='')
        elif self.ID in ['A01', 'A02']:
            print('{:>8}'.format('2C'), end='')
        else:
            print('{:>8}'.format('C'), end='')
        
        print('\tquat\thw_vec')
        print(72*'—')

        # Loop through irreps and display if in the range for H
        for irrep in self.irreps:
            if irrep['H'] >= H_min and irrep['H'] <= H_max:
                print('{:15}{:6}{:8}{:8}{:8}\t{}\t{}'.format(*irrep))

    def generate_irreps(self, H_max):
        """Creates all irreps for this group of bounded dimension.

        Note that H_R = dim(R) except for quaternionic irreps free of Witten anomalies,
        in which case H_R = dim(R)/2. A complete list of irreps with H ≤ H_max is found
        by using the Weyl dimension formula and backtracking on the highest-weight vector.

        H_max is replaced with max(H_max, V=dim(adj)) to ensure irreps up to the adjoint
        are created (this ensures all irreps which appear in type-B vertices are present).

        Args:
            H_max (int): Upper bound on H_R.

        """

        # Go out to atleast the adjoint
        H_max = max(H_max, self.V)

        # If there are quaternionic irreps, actually go out to dim(R) ≤ 2*H_max
        # and then at the end restrict to H_R ≤ H_max
        if self.has_quaternionic:
            H_max *= 2

        # Arrays for storing highest-weight vectors and corresponding data
        hw_vec_list = np.empty([0],    dtype='str')
        hw_vec_data = np.empty([0, 5], dtype='int')

        # Now use backtracking on the highest-weight vector
        hw_vec = np.zeros(self.n, dtype='int')
        ii = 0

        done = False
        while not done:
            # Get data for current highest-weight vector
            irrep_data = self.get_irrep_data(hw_vec)
            
            if irrep_data[0] <= H_max:
                # Comes in below the bound: add to list and increment to next vector
                hw_vec_str = '(' + ','.join([str(d) for d in hw_vec]) + ')'
                hw_vec_list = np.append(hw_vec_list, [hw_vec_str], axis=0)
                hw_vec_data = np.append(hw_vec_data, [irrep_data], axis=0)

                hw_vec[0] += 1
                ii = 0

            elif ii == len(hw_vec) - 1:
                # End of backtracking
                done = True

            else:
                # Roll-over: dimension is too high
                ii += 1
                hw_vec[ii] += 1
                hw_vec[:ii] = 0

        # Remove redundant (conjugate) irreps
        unique, inverse = np.unique(hw_vec_data[:,:4], axis=0, return_index=True)
        hw_vec_data = hw_vec_data[inverse]
        hw_vec_list = hw_vec_list[inverse]

        # Use H, A and hw_vec to order irreps (this helps latin labels be in
        # a somewhat natural order, e.g. vector of SO(8) is 'a' while spinor is 'b')
        data_for_sort = np.array([[data[0], data[1], *[-int(xx) for xx in hw[1:-1].split(',')]]
                                    for data, hw in zip(hw_vec_data, hw_vec_list)])
        order = np.lexsort(data_for_sort.T[::-1])
        hw_vec_data = hw_vec_data[order]
        hw_vec_list = hw_vec_list[order]

        # Store the irrep data
        irreps = np.empty(len(hw_vec_data), dtype=dt_irrep)
        irreps['H']      = hw_vec_data[:, 0]
        irreps['A']      = hw_vec_data[:, 1]
        irreps['B']      = hw_vec_data[:, 2]
        irreps['C']      = hw_vec_data[:, 3]
        irreps['quat']   = hw_vec_data[:, 4]
        irreps['hw_vec'] = list(hw_vec_list)

        # Build latin indices for distinguishing irreps of the same dimension

        # Find groups of irreps with the same value of dim(R)
        duplicates, inverse, counts = np.unique(irreps['H'], return_inverse=True, return_counts=True)

        # Populate an array of latin indices (entry will remain empty if irrep of that dimension is unique)
        latin_indices = np.array(len(irreps) * [''])
        for jj, count in enumerate(counts):
            if count > 1:
                whr = np.where(inverse == jj)[0]
                for kk, ind in enumerate(whr):
                    latin_indices[ind] = chr(kk + 97)   # a,b,c,...

        # Set irrep IDs: irr-[group ID]-[H] or irr-[group ID]-[H]-[latin]
        for irrep, latin in zip(irreps, latin_indices):
            irrep['ID'] = 'irr-{}{}-{}{}'.format(self.cartan_type, self.n, irrep['H'], latin)

        # Identify quaternionic irreps free of Witten anomalies and update data
        # Put a '-h' tag on the ID to indicate this is a (h)alf-hypermultiplet
        for irrep in irreps:
            if irrep['quat'] and (irrep['A'] % 2 == 0) and self.ID != 'A01':
                irrep['H'] /= 2
                irrep['A'] /= 2
                irrep['B'] /= 2
                irrep['C'] /= 2
                irrep['ID'] += '-h'

        if self.has_quaternionic:
            # Clean up irreps to only those with H ≤ "Hmax original"
            irreps = irreps[irreps['H'] <= H_max//2]

        # Remove trivial irrep
        irreps = irreps[1:]

        # Sort by H and save
        order = np.argsort(irreps['H'])
        self.irreps = irreps[order]

    def get_irrep_data(self, hw_vec):
        """Returns information about an irrep with highest-weight vector `hw_vec`.

        The data returned are the dimension, computed using the Weyl dimension formula,
        the trace indices A, B and C (computed using the results of [Okubo 1982] adjusting
        for our conventions) and a boolean for if the irrep is quaternionic (see `is_quaternionic()`).

        For A[1] and A[2], C can be a half-integer so 2*C is returned instead (this is accounted for when
        creating vertices). Similarly, for B[3] and D[4] C is in (1/4)Z and so C + 3B/4 (an integer)
        is returned in its place (the sum over C is unaffected when the B-constraint is satisfied).

        Args:
            hw_vec (list): Integer array giving the highest-weight vector.

        Returns:
            list: Array of data containing information about the irrep R:
                    (dim(R), A, B, C, quat), where the first four are integers and the last is a boolean.

        """

        if max(hw_vec) == 0:
            # Trivial irrep
            return 1, 0, 0, 0, False

        # Based on [Okubo '82], including equation references.
        # Each Cartan type preprocesses highest-weight vector differently because
        # of different Cartan matrix structures.

        # Following the ref, we write m_j for the non-negative integer components
        # of the heighest-weight vector. We use sigma (called 'l' for B,C,D in the paper)
        # to label the half-integers in terms of which quadratic and quartic invariants are calculated

        no_quartic = False

        if self.cartan_type == 'A':

            # (3.10), note N=n+1 and j=1,2,...,(n+1)
            sigma_0 = (self.n + 2)/2 - np.arange(1, self.n+2)

            # Compute fj from mj as cumulative sum then subtract off mean in (3.6)
            fj = np.array([*np.cumsum(hw_vec[::-1])[::-1], 0])
            sigma = sigma_0 + fj - np.mean(fj)

            # (3.9)
            coeff_I2 = 1

            # (3.12)
            coeff_J4_a = (self.V + 2)
            coeff_J4_b = (2*self.V - 1)/(self.n+1)

            if self.n in [1, 2]:
                no_quartic = True

        elif self.cartan_type == 'B':

            # (3.15) and (3.16)
            sigma_0 = self.n - np.arange(1, self.n+1) + 1/2
            fj = [sum(hw_vec[jj:]) - hw_vec[-1]/2 for jj in range(self.n)]
            sigma = sigma_0 + fj
            
            # (3.17a)
            coeff_I2 = 1

            # (3.17c)
            coeff_J4_a = (2*self.n**2 + self.n + 2)/8
            coeff_J4_b = (4*self.n + 1)/8

        elif self.cartan_type == 'C':
            
            # (3.21)
            sigma_0 = self.n - np.arange(1, self.n+1) + 1
            fj = [sum(hw_vec[jj:]) for jj in range(self.n)]
            sigma = sigma_0 + fj
            
            # (3.22a)
            coeff_I2 = (1/2)

            # (3.23)
            coeff_J4_a = (2*self.n**2 + self.n + 2)/8
            coeff_J4_b = (4*self.n + 1)/8

        elif self.cartan_type == 'D':

            # (3.25) and (3.26)
            sigma_0 = self.n - np.arange(1, self.n+1)
            fj = [sum(hw_vec[jj:-2]) + (hw_vec[-2] + hw_vec[-1])/2 for jj in range(self.n-2)]
            fj = np.array([*fj, (hw_vec[-2] + hw_vec[-1])/2, (-hw_vec[-2] + hw_vec[-1])/2])
            sigma = sigma_0 + fj
            
            # (3.27a)
            coeff_I2 = 1

            # (3.28)
            coeff_J4_a = (2*self.n**2 - self.n + 2)/8
            coeff_J4_b = (4*self.n - 1)/8

        elif self.cartan_type == 'E':
            no_quartic = True

        elif self.cartan_type == 'F':
            no_quartic = True

        elif self.cartan_type == 'G':
            no_quartic = True

        # Compute irrep dimension
        if self.cartan_type in ['A', 'B', 'C', 'D']:
            num, den = 1, 1

            if self.cartan_type in ['B', 'C']:
                # These have an additional product over sigma/sigma_0
                for ii in range(len(sigma)):
                    num *= int(np.round(2*sigma[ii]))
                    den *= int(np.round(2*sigma_0[ii]))

                    gcd = GCD(num, den)
                    num = int(num // gcd)
                    den = int(den // gcd)

            for ii, jj in combinations(range(len(sigma)), 2):
                
                if self.cartan_type == 'A':
                    # (eqn right before section 3.B)
                    num *= int(np.round(2*(sigma[ii]   - sigma[jj]  )))
                    den *= int(np.round(2*(sigma_0[ii] - sigma_0[jj])))

                elif self.cartan_type in ['B', 'C', 'D']:
                    # (eqns at ends of sections 3.B, 3.C and 3.D)
                    num *= int(np.round(2*(sigma[ii]**2   - sigma[jj]**2  )))
                    den *= int(np.round(2*(sigma_0[ii]**2 - sigma_0[jj]**2)))

                # Reduce at each step to keep numerator and denominator
                # relatively prime (and thus relatively small)
                gcd = GCD(num, den)
                num = int(num // gcd)
                den = int(den // gcd)

            # If everything has gone correctly the dimension is an integer (i.e. the denominator is 1)
            dim_R = num

            # Check whether something has gone wrong
            if num < 0 or den != 1:
                err = 'Irrep dimension is rational or negative:\n'
                err += f'        group: {self.ID}\n'
                err += f'    hw vector: {hw_vec}\n'
                err += f'    numerator: {num}\n'
                err += f'  denominator: {den}\n'
                raise Exception(err)

            # Quadratic Casimir
            index_I2 = coeff_I2 * sum(sigma**2 - sigma_0**2)

            # Predictable rational denominator: split so that can use integer arithmetic later
            if self.cartan_type == 'A':
                index_I2_denominator = (self.n+1)
            elif self.cartan_type == 'B':
                index_I2_denominator = 4
            elif self.cartan_type == 'C':
                index_I2_denominator = 2
            elif self.cartan_type == 'D':
                index_I2_denominator = 4
            
            index_I2_numerator = int(np.round(index_I2_denominator*index_I2))

        else:
            # Exceptionals
            # Use Weyl dimension formula directly using simple roots

            if   self.ID == 'E06': roots = ROOTS_E6
            elif self.ID == 'E07': roots = ROOTS_E7
            elif self.ID == 'E08': roots = ROOTS_E8
            elif self.ID == 'F04': roots = ROOTS_F4
            elif self.ID == 'G02': roots = ROOTS_G2

            num = 1
            den = 1
            for root in roots:
                num *= int(root @ (hw_vec + 1))
                den *= int(sum(root))

                gcd = GCD(num, den)
                num = int(num // gcd)
                den = int(den // gcd)
            dim_R = num // den

            if self.cartan_type == 'E':
                if self.n == 6:
                    inner_prod = (1/3) * np.array([
                        [ 4,  5,  6,  4,  2,  3],
                        [ 5, 10, 12,  8,  4,  6],
                        [ 6, 12, 18, 12,  6,  9],
                        [ 4,  8, 12, 10,  5,  6],
                        [ 2,  4,  6,  5,  4,  3],
                        [ 3,  6,  9,  6,  3,  6]
                    ])

                if self.n == 7:
                    inner_prod = (1/2) * np.array([
                        [ 3,  4,  5,  6,  4,  2,  3],
                        [ 4,  8, 10, 12,  8,  4,  6],
                        [ 5, 10, 15, 18, 12,  6,  9],
                        [ 6, 12, 18, 24, 16,  8, 12],
                        [ 4,  8, 12, 16, 12,  6,  8],
                        [ 2,  4,  6,  8,  6,  4,  4],
                        [ 3,  6,  9, 12,  8,  4,  7]
                    ])
                    
                if self.n == 8:
                    inner_prod = np.array([
                        [ 4,  7, 10,  8,  6,  4,  2,  5],
                        [ 7, 14, 20, 16, 12,  8,  4, 10],
                        [10, 20, 30, 24, 18, 12,  6, 15],
                        [ 8, 16, 24, 20, 15, 10,  5, 12],
                        [ 6, 12, 18, 15, 12,  8,  4,  9],
                        [ 4,  8, 12, 10,  8,  6,  3,  6],
                        [ 2,  4,  6,  5,  4,  3,  2,  3],
                        [ 5, 10, 15, 12,  9,  6,  3,  8]
                    ])

            elif self.cartan_type == 'F':
                inner_prod = (1/2) * np.array([
                    [ 2,  3,  4,  2],
                    [ 3,  6,  8,  4],
                    [ 4,  8, 12,  6],
                    [ 2,  4,  6,  4]
                ])

            elif self.cartan_type == 'G':
                inner_prod = (1/3) * np.array([
                    [2, 3],
                    [3, 6]
                ])
        
            index_I2 = hw_vec @ inner_prod @ (hw_vec + 2)

            if self.cartan_type == 'E' and self.n == 6:
                index_I2_denominator = 3
            elif self.cartan_type == 'E' and self.n == 7:
                index_I2_denominator = 2
            elif self.cartan_type == 'E' and self.n == 8:
                index_I2_denominator = 1
            elif self.cartan_type == 'F':
                index_I2_denominator = 1
            elif self.cartan_type == 'G':
                index_I2_denominator = 3

            index_I2_numerator = int(np.round(index_I2_denominator*index_I2))

        # Quadratic index
        index_A = (index_I2_numerator * dim_R) // (index_I2_denominator * self.V)

        # Quartic indices        
        K_R_numerator = self.V * (6*index_I2_numerator - self.A_adj * index_I2_denominator)
        K_R_denominator = 2*(self.V+2) * dim_R * index_I2_numerator

        if no_quartic:
            # B = 0 and C is calculated differently.
            index_B = 0
            index_C = (K_R_numerator * index_A**2) / K_R_denominator

        else:
            # Quartic Casimir
            index_J4 = coeff_J4_a * sum(sigma**4) - coeff_J4_b * sum(sigma**2)**2 + self.V*self.J4_adj/240

            if self.cartan_type == 'A':
                index_J4_numerator = int(np.round((self.n+1)*index_J4))
                index_J4_denominator = (self.n+1)
            else:
                index_J4_numerator = int(np.round(480*index_J4))
                index_J4_denominator = 480

            if self.ID == 'D04':
                # Use vector irrep as reference irrep, since J4(adj) = 0 for D4
                J4_fund = 1575 / 8
                B_fund = 4
                CmKA2_fund = -1

                ratio = dim_R * index_J4 / (8 * J4_fund)
                index_B = ratio * B_fund
                index_C = ratio * CmKA2_fund + (K_R_numerator * index_A**2) / K_R_denominator

            else:
                # From solving (1.38) after subbing in definitions of B,C in Tr(X^4).
                ratio_numerator = dim_R * index_J4_numerator
                ratio_denominator = self.V * int(np.round(index_J4_denominator * self.J4_adj))

                index_B = (ratio_numerator * self.B_adj) / ratio_denominator

                index_C_numerator = ratio_numerator * self.CmKA2_adj_numerator * K_R_denominator \
                                    + K_R_numerator * index_A**2 * ratio_denominator * self.CmKA2_adj_denominator
                index_C_denominator = ratio_denominator * self.CmKA2_adj_denominator * K_R_denominator
                index_C = index_C_numerator / index_C_denominator

        is_quaternionic = self.is_quaternionic(hw_vec)

        # For these groups C is not an integer: store instead C+3B/4, which *is*
        # an integer and does not change sums of C when the B-constraint is satisfied
        if self.ID in ['B03', 'D04']:
            index_C += 3*index_B/4

        # For these groups C is not an integer: store 2C, which *is* and integer, instead
        if self.ID in ['A01', 'A02']:
            index_C *= 2

        # Double check A, B and C (or "C"=C+3B/4 or "C"=2C) are integers within some threshold
        if abs(index_A - round(index_A)) > 10**-12: print('A is not an integer:', index_A)
        if abs(index_B - round(index_B)) > 10**-12: print('B is not an integer:', index_B)
        if abs(index_C - round(index_C)) > 10**-12: print('C is not an integer:', index_C)

        # Correct fp errors
        index_A = int(round(index_A))
        index_B = int(round(index_B))
        index_C = int(round(index_C))

        return dim_R, index_A, index_B, index_C, is_quaternionic
    
    def is_quaternionic(self, hw_vec):
        """Returns whether the irrep with highest-weight vector `hw_vec` is quaternionic.

        Quaternionic irreps exist only for the following groups:
            - A[n] ~ SU(n+1) with n ≡ 1 mod 4
            - B[n] ~ SO(2n+1) with n ≡ 1,2 mod 4
            - C[n] ~ Sp(n) for all n
            - D[n] ~ SO(2n) for n ≡ 2 mod 4
            - E[7]

        Args:
            hw_vec (list): Integer array giving the highest-weight vector.

        Returns:
            bool: Whether the irrep is quaternionic.

        """

        if self.cartan_type == 'A' and (self.n % 4 == 1):
            # For A[n] ~ SU(n+1), quaternionic irreps appear only for n = 1 mod 4
            # and are exactly those irreps for which the highest-weight vector
            # is symmetric and has odd central value
            symmetric = np.all(hw_vec == hw_vec[::-1])
            center_value = hw_vec[(self.n-1)//2]
            
            return symmetric and (center_value % 2 == 1)
        
        elif self.cartan_type == 'B' and (self.n % 4 in [1, 2]):
            # For B[n] ~ SO(2n+1), quaternionic irreps appear only
            # for n = 1 and n = 2 mod 4 and are those irreps for which
            # the last entry (corresponding to the short root)
            # of the highest-weight vector is odd
            return hw_vec[-1] % 2 == 1
        
        elif self.cartan_type == 'C':
            # For C[n] ~ Sp(n), quaternionic irreps are exactly those for which
            # the sum over odd components of the highest-weight vector is odd
            return sum(hw_vec[::2]) % 2 == 1
        
        elif self.cartan_type == 'D' and (self.n % 4 == 2):
            # For D[n] ~ SO(2n), quaternionic irreps only appear for n = 2 mod 4
            # and are those for which the last two entries of the highest-weight
            # vectors are opposite parity
            return (hw_vec[-2] + hw_vec[-1]) % 2 == 1

        elif self.cartan_type == 'E' and self.n == 7:
            # The only exceptional group with quaternionic irreps is E7 and these
            # are the irreps for which a1+a3+a7 is odd (note zero-indexing)
            return (hw_vec[0] + hw_vec[2] + hw_vec[6]) % 2 == 1

        return False

    def reset_AmaxG(self, vertex, min_irrep_H):
        self.AmaxG     = [0]
        self.AmaxG_hat = [0]

        self.vertex_HAs = [[irr['H'], irr['A']] for irr in vertex.irreps if irr['n'] >= min_irrep_H]

    def get_AmaxG(self, n):
        """Computes and returns the value of Amax(n), using memoization.

        The function Amax is defined to be Amax(n) = max{ sum(nR*AR) | sum(nR*HR) ≤ n },
        and is computed recursively using Amax_hat (see get_AmaxG_hat())
            • Amax(n=0) = 0
            • Amax(n>0) = max{ Amax(n-1), Amax_hat(n) }
        In contrast to Amax_hat, Amax(n) is always finite.

        Args:
            n (int): The upper bound for sum(nR*HR).

        Returns:
            (int): Amax(n), the maximum value for sum(nR*AR) given the constraint sum(nR*HR)≤n.

        """

        # Check if value exists and return if it does
        while len(self.AmaxG) <= n:
            self.AmaxG.append(None)
        if self.AmaxG[n] is not None:
            return self.AmaxG[n]

        # Otherwise, compute it, save it and return
        value = max(self.get_AmaxG(n-1), self.get_AmaxG_hat(n))
        self.AmaxG[n] = value

        return value   
    
    def get_AmaxG_hat(self, n):
        """Computes and returns the value of Amax_hat(n), using memoization.

        The function Amax_hat is defined to be Amax_hat(n) = sup{ sum(nR*AR) | sum(nR*HR) = n },
        and is computed recursively as
            • Amax_hat(n<0) = -∞
            • Amax_hat(n=0) = 0
            • Amax_hat(n>0) = sup_R{ Amax_hat(n-HR) + AR }
        This function is used as an intermediate step to compute Amax (see `get_AmaxG()`).

        Args:
            n (int): Value for sum(nR*HR).

        Returns:
            (int): Amax_hat(n), the supremum value for sum(nR*AR) given the constraint sum(nR*HR)=n.
                    If sum(nR*HR)=n is unobtainable, then Amax_hat(n) = sup{∅} = -np.inf.

        """

        # Check if value exists and return if it does
        while len(self.AmaxG_hat) <= n:
            self.AmaxG_hat.append(None)
        if self.AmaxG_hat[n] is not None:
            return self.AmaxG_hat[n]

        # Otherwise, identify irreps with HR ≤ n and compute supremum
        value = -np.inf
        for H_v, A_v in self.vertex_HAs:
            if H_v <= n:
                value = max(value, self.get_AmaxG_hat(n - H_v) + A_v)

        # Save it and return
        self.AmaxG_hat[n] = value

        return value

    def get_Hmin_bounds(self, H_max, progress=True):
        """Computes the sequence H_min(B;j), the minimal sum(nR*HR) needed to acheive sum(nR*BR)=B
        using one additional irrep at each step.

        If the irreps are sorted in order of increasing |B|/H, H_min(B;j) gives the minimal
        value of sum(nR*HR) for which sum(nR*BR)=B, using the first j irreps. Note that H_min(B=0;0) = 0
        and H_min(B≠0;0) = ∞. This sequence of bounds is used in `get_HB_solutions()` when adding
        irreps one-by-one to solve the B-constraint.

        Args:
            H_max (int): Upper bound on sum(nR*HR).

        Returns:
            B_min_sequence (list): Integer array of B off-sets. The j-th value gives
                                    the off-set of the indexing of H_min(B;j).
            H_min_sequence (list): 2d array containing the sequence of bounds H_min(B;j).
                                    The j-th row is the bound H_min(B) using the j irreps of smallest |B|/H.
            subset (list): indices of irreps with H ≤ `H_max` in order of decreasing |B|/H

        """

        # Identify the subset of irreps with H ≤ H_max
        subset = np.where(self.irreps['H'] <= H_max)[0]

        if len(subset) == 0:
            # No irreps small enough
            return [0], [[0]], []

        # Compute the ratios B/H for the irreps and the min/max values of B which
        # could ever appear in the following computation, with a buffer to be safe
        ratios = self.irreps[subset]['B'] / self.irreps[subset]['H']
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        B_min = min(int(np.round(min_ratio * H_max)), 0) - B_PAD
        B_max = max(int(np.round(max_ratio * H_max)), 0) + B_PAD
        
        # Sort the irreps in order of increasing |B|/H
        order = np.argsort(abs(ratios))
        subset = subset[order]

        # Create array for storing the H bounds
        # The value corresponding to B lies at index ii=B-B_min
        # If a value B is unacheivable then H_bound(B) = H_max + 1 > H_max
        H_bounds = (H_max+1) * np.ones(B_max - B_min, dtype='int')

        # Initialize with empty configuration: (B,H)=(0,0) is acheivable
        H_bounds[0 - B_min] = 0

        # Initialize array for storing sequence of bounds
        H_bounds_sequence = []

        # Loop over irrep (B,H) data
        iterator = self.irreps[subset]
        if progress:
            iterator = tqdm(iterator, desc=f'  {self.ID} building H bounds', ascii=True, leave=False)

        for irrep in iterator:

            # Save current bounds to sequence
            H_bounds_sequence = [H_bounds.copy(), *H_bounds_sequence]

            # Add to (B,H)=(0,0) until a value of B which has already
            # been acheived is hit or H_max is hit
            n_max = H_max // irrep['H']

            for nn in range(1, n_max + 1):
                B_new = nn * irrep['B']
                H_new = nn * irrep['H']

                if H_new < H_bounds[B_new - B_min]:
                    H_bounds[B_new - B_min] = H_new
                else:
                    n_max = nn
                    break

            # Now scan through current bounds adding up to n_max of the current irrep
            for ii, H_bound in enumerate(H_bounds):
                for nn in range(1, n_max + 1):
                    ii_new = ii + nn * irrep['B']
                    H_new = H_bound + nn * irrep['H']

                    if ii_new < 0 or ii_new >= len(H_bounds) or H_new >= H_bounds[ii_new]:
                        break
                    else:
                        H_bounds[ii_new] = H_new

        # Trim sequence of bounds
        B_min_sequence = []
        for ii, H_bounds in enumerate(H_bounds_sequence):
            whr = np.where(H_bounds <= H_max)[0]
            index_min = whr[0] - B_PAD
            index_max = whr[-1] + B_PAD
            H_bounds_sequence[ii] = H_bounds[index_min:index_max]
            B_min_sequence.append(B_min + index_min)

        # Return sequence of bounds and subset of irreps in order of *decreasing* |B|/H
        return B_min_sequence, H_bounds_sequence, subset[::-1]

    def get_vertices(self, H_max, T_min_max, progress=True):
        """Returns all vertices (with H ≤ `H_max` for type-A) which have T_min ≤ `T_min_max`.

        Args:
            H_max (int): Upper bound on H.
            T_min_max (int): Upper bound on T_min.
            progress (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            list: Array of vertices, sorted by H.

        """
        
        # Find all solutions to the B-constraint
        n_solns = self.get_B_constraint_solutions(H_max, self.B_adj, progress)

        # Loop through creating vertex objects
        group_vertices = []

        iterator = n_solns
        if progress:
            iterator = tqdm(iterator, desc=f'  {self.ID} building vertices', ascii=True, leave=False)

        for n_soln in iterator:
            # Create vertex object and save
            vertex_new = Vertex(self, n_soln)
            
            if vertex_new.T_min <= T_min_max:
                group_vertices.append(vertex_new)

        # Check for vertex ID collisions and differentiate them
        ids, inverse, counts = np.unique([v.ID for v in group_vertices], return_inverse=True, return_counts=True)

        for ii, count in enumerate(counts):
            if count > 1:
                # Multiple instances of the same ID
                for jj, index in enumerate(np.where(inverse == ii)[0]):
                    # Get fixed number of base-26 digits (with leading zeros)
                    digs26 = []
                    while len(digs26) < np.log(count)/np.log(26):
                        digs26 = [jj % 26, *digs26]
                        jj = jj//26

                    # Append corresponding latin letters
                    # e.g. [5] → 'f', [9, 3] → 'jd', [0, 2, 1] → 'acb'
                    latin = ''.join([chr(dd+97) for dd in digs26])
                    group_vertices[index].ID += '-' + latin

        return group_vertices

    def get_B_constraint_solutions(self, H_max, B_target, progress=True):
        """Returns all irrep multiplicities which solve sum(nR*BR) = `B_target` and have sum(nR*HR) ≤ `H_max`.

        Solutions are built by adding in irreps one-by-one in order of decreasing |B|/H.
        A list of irrep multiplicities with sum(nR*HR) ≤ `H_max` is maintained. At step j, any number
        of irrep j are added to the configurations using irreps <j while respecting the upper bounds
        H_ub(B;j) which depend on the current value of sum(nR*BR). The point is that because of the order
        in which the irreps are incorporated, the upper bounds on H to acheive sum(nR*BR) = `B_target`
        before violating sum(nR*HR) ≤ `H_max` quickly tighten on the target B-sum.

        Args:
            H_max (int): Upper bound on sum(nR*HR)
            B_target (int): Target value for sum(nR*BR)

        Returns:
            list: 2d array with each row being a set of irrep multiplicities which solve
                    sum(nR*BR) = `B_target` subject to sum(nR*HR) ≤ `H_max`

        """

        # Check some edge cases
        if H_max <= 0:
            if H_max == 0 and B_target == 0:
                # Only the empty configuration works
                return [0], [np.zeros(len(self.irreps), dtype='int')]
            else:
                # No configurations exist
                return [], []

        # Get sequence of H_min(B;j) bounds and the corresponding order of irreps
        bound_data = self.get_Hmin_bounds(H_max, progress)
        B_min_array, H_bounds_array, subset = bound_data

        # Transform Hmin bounds to sequence of upper bounds
        H_ub_array = [H_max - H_bounds[::-1] for H_bounds in H_bounds_array]
        B_ub_min_array = [B_target - B_min - len(H_ub) + 1 for B_min, H_ub in zip(B_min_array, H_ub_array)]

        # This solver object keeps track of potentially viable irrep multiplicities
        # and has methods for adding, bounding and pruning potential solutions
        solutions = ConstraintSolver(len(self.irreps))

        # Iterator over irreps in order of decreasing |B|/H, adding to current
        # partial solutions and removing solutions by using the upper bounds
        iterator = zip(subset, B_ub_min_array, H_ub_array)
        if progress:
            iterator = tqdm(iterator, total=len(subset), ascii=True, desc=f'  {self.ID} adding irreps', leave=False)

        for index, B_ub_min, H_ub in iterator:
            solutions.add_irrep(index, self.irreps[index], B_ub_min, H_ub)
            solutions.remove_solutions(B_ub_min, H_ub)

        # Extract solutions and ensure the list of solutions includes all of those
        # of type B solutions (bi.bi ≤ 0), generated separately
        ns = solutions.soln_n[0]
        ns_typeBC = self.get_type_B_solutions()
        ns = np.append(ns, ns_typeBC, axis=0)
        ns = np.unique(ns, axis=0)

        # Sort the solutions by sum(nR*HR)
        soln_H = ns @ self.irreps['H']
        order = np.argsort(soln_H)
        ns = ns[order]

        if self.ID == 'G02':
            # For G[2], restrict to solutions with no global anomaly (sum(nR*CR) - Cadj ≡ 0 mod 3)
            Csums = ns @ self.irreps['C'] - self.C_adj
            anomfree = (Csums % 3 == 0)
            ns = ns[anomfree]

        elif self.ID == 'A01':
            # For A[1], 'C' = 2C = integer and need sum(nR*CR) - Cadj ≡ 0 mod 12
            Csums = ns @ self.irreps['C'] - self.C_adj
            anomfree = (Csums % (2*12) == 0)
            ns = ns[anomfree]
            
        elif self.ID == 'A02':
            # For A[1], 'C' = 2C = integer and need sum(nR*CR) - Cadj ≡ 0 mod 6
            Csums = ns @ self.irreps['C'] - self.C_adj
            anomfree = (Csums % (2*6) == 0)
            ns = ns[anomfree]

        return ns

    def get_type_B_solutions(self):
        """Returns irrep multiplicities for all type-B vertices (i.e. with bi.bi ≤ 0).

        Returns:
            list: 2d array with each row being the multiplicities of irreps giving a solution
                    of the B-constraint which additionally has bi.bi ≤ 0.

        """

        # Identify irreps with C <= C_adj
        inds = np.where(self.irreps['C'] <= self.C_adj)[0]

        # Sort so irrep with C=0 (if any) is first,
        # otherwise the backtracking gets stuck
        order = np.argsort(self.irreps['C'][inds])
        inds = inds[order]

        # Array for solutions with sum(C) <= C_adj
        ns_typeBC = []

        # Use backtracking to find all solutions
        ns = np.zeros(len(self.irreps), dtype='int')
        ns[inds[0]] = 1 # Start with nontrivial solution
        ii = 0

        done = False
        while not done:

            Bsum = ns @ self.irreps['B']
            Csum = ns @ self.irreps['C']

            if Csum <= self.C_adj:
                # Good: will satisfy bi.bi ≤ 0 bound.
                
                # Only save if the B-constraint is satisfied
                if Bsum == self.B_adj:
                    ns_typeBC.append(ns.copy())

                if ii == 0 and self.irreps['B'][inds[0]] * (self.B_adj - Bsum) < 0:
                    # If the ii=0 irrep has B of the 'wrong' sign (so adding more moves
                    # farther from satisfying the B-constraint), roll over
                    ii += 1
                    ns[inds[ii]] += 1
                    ns[inds[:ii]] = 0

                else:
                    # Otherwise, increment
                    ii = 0
                    ns[inds[0]] += 1

            elif ii == len(inds) - 1:
                # Backtracking is done
                done = True

            else:
                # Roll-over
                ii += 1
                ns[inds[ii]] += 1
                ns[inds[:ii]] = 0

        return ns_typeBC
    
    def save_irreps(self, folder):
        """Saves irrep data in a '.tsv' file.

        The created file is "[`folder`]/[group_ID]-irreps.tsv".

        Args:
            folder (str): Path to folder where file is to be saved.

        """

        filepath = folder + f'/{self.ID}-irreps.tsv'

        # If file exists, check if already contains all irreps currently present
        H_max_saved = 0
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                for line in file.readlines():
                    H_max_saved = max(H_max_saved, int(line.split('\t')[1]))

        if max(self.irreps['H']) > H_max_saved:
            # Create/overwrite file
            with open(filepath, 'w') as file:
                for irr in self.irreps:
                    irr_string = '\t'.join([str(xx) for xx in irr])
                    file.write(irr_string + '\n')

def GCD(x, y):

    if y == 0:
        return abs(x)
    
    return GCD(y, x % y)
