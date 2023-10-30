import numpy as np

from edge import Edge
from gramMatrix import get_T_min
from irreducibleClique import IrreducibleClique


# (d)ata(t)ype for storing irreps' data: multiplicity, irrep ID, etc.
dt_irrep_mults = np.dtype([('n', 'int'), ('irr_ID', 'str', 255), ('H', 'int'), ('A', 'int')])

# (d)ata(t)ype for storing product irreps: multiplicity and irrep ID
dt_prod_irrep = np.dtype([('n', 'int'), ('irr_ID', 'str', 255)])


class Vertex():

    def __init__(self, group, n_soln):
        """Initializes a vertex associated to a simple theory with gauge group `group` that solves the B-constraint.

        Args:
            group (SimpleGroup): Simple group of the simple theory.
            n_soln (array): Array of non-negative integers giving the multiplicities of irreps of `group`
                            which solves the B-constraint.

        """

        # Δ = H-V and Gram matrix entries
        self.Δ    =  (n_soln @ group.irreps['H']) - group.V
        self.b0bi = ((n_soln @ group.irreps['A']) - group.A_adj) // 6
        self.bibi = ((n_soln @ group.irreps['C']) - group.C_adj) // 3

        if group.ID in ['A01', 'A02']:
            # For these groups 'C' = 2C = integer
            self.bibi = self.bibi // 2

        # Vertex type based on Gram matrix entry bi.bi
        if self.bibi > 0: self.type = 'A'
        else:             self.type = 'B'

        # Irrep multiplicities and relevant data
        irreps = [(n, irr['ID'], irr['H'], irr['A']) for n, irr in zip(n_soln, group.irreps) if n > 0]
        self.irreps = np.array(irreps, dtype=dt_irrep_mults)
        self.irreps_string = ' + '.join([str(n) + ' x ' + '(' + id + ')' for n, id, H, A in self.irreps])

        # Create an ID of the form vtx-[group ID]-[Δ]-[bibi]-[b0bi]
        # This is not guaranteed to be unique: collisions are fixed when vertices
        # are created by the parent group and disambiguated by appending -[latin]
        # Use 'n' to indicate negative values
        
        self.ID = f'vtx-{group.ID}-'

        if self.Δ < 0:
            self.ID += 'n'
        self.ID += str(abs(self.Δ)) + '-'

        if self.bibi < 0:
            self.ID += 'n'
        self.ID += str(abs(self.bibi)) + '-'
        
        if self.b0bi < 0:
            self.ID += 'n'
        self.ID += str(abs(self.b0bi))

        # Get T_min
        self.T_min = get_T_min(np.array([self.b0bi]), np.array([[self.bibi]]))

        # Arrays which will store nontrivial edges to vertices of type-A and type-B
        self.edges_nontrivial_A = []
        self.edges_nontrivial_B = []

        # For keeping track of whether this vertex appears in bounded/anomaly-free cliques
        self.used = False

    def __lt__(self, other):

        # Define '<' for vertices by comparing...
        
        # type (type B before type A)
        if self.type != other.type:
            return self.bibi < other.bibi

        # Δ = H-V (smallest first)
        elif self.Δ != other.Δ:
            return self.Δ < other.Δ
        
        # Gram matrix entry bi.bi (smallest first)
        elif self.bibi != other.bibi:
            return self.bibi < other.bibi
        
        # Gauge group, fetched from ID (alphabetically then by rank, e.g. A03 < A04 < B02)
        elif self.ID[4:7] != other.ID[4:7]:
            return self.ID[4:7] < other.ID[4:7]
        
        # ID, which is necessarily unique
        else:
            return self.ID < other.ID

    def display(self):
        """Outputs information about the vertex."""

        print(f'{self.ID:20}   Δ = {self.Δ:4}    bi.bi = {self.bibi:4}    ' \
                + f'b0.bi={self.b0bi:4}    H = {self.irreps_string}')

    def get_save_string(self):
        """Returns a tsv string containing information about the vertex."""

        data_string = self.ID
        data_string += f'\tΔ = {self.Δ}'
        data_string += f'\tbi.bi = {self.bibi}'
        data_string += f'\tb0.bi = {self.b0bi}'
        data_string += '\t' + self.irreps_string

        return data_string

    def add_gram_entries(self, clique, ii):
        """Adds the vertex's bi.bi and b0.bi to the gram matrix data of `clique` in given row/column index `ii`."""
        
        clique.gram_bibj[ii, ii] = self.bibi
        clique.gram_b0bi[ii]     = self.b0bi

    def construct_edges(self, other):
        """Constructs and stores all non-trivial edges between two vertices.

        This is done in two steps:
            - All possible product irreps are collected: nR x (R) and nS x (S)
                can be merged to (R,S) only if nS ≥ HR and nR ≥ HS.
            - All combinations of product irreps are found which are admissible don't overuse
                irreps from either vertex. This is done using backtracking.
        Constructed edges are stored in both vertices' edge_nontrivial_A/B arrays,
        as appropriate.

        Args:
            other (Vertex): The vertex at the head of edges from `self`.
        
        """


        # First find all possible product irreps
        irrep_pairs = []

        for ii, irr0 in enumerate(self.irreps):
            for jj, irr1 in enumerate(other.irreps):

                if irr0['n'] < irr1['H'] or irr1['n'] < irr0['H']:
                    # Irrep multiplicities are too small to merge
                    continue

                # n0 and n1 are arrays which store how many of each irrep are used
                n0 = np.zeros(len(self.irreps), dtype='int')
                n0[ii] = irr1['H']
                
                n1 = np.zeros(len(other.irreps), dtype='int')
                n1[jj] = irr0['H']

                # Compute the H discount (δH < 0) and off-diagonal Gram matrix entry contribution
                δH = -irr0['H'] * irr1['H']
                AA =  irr0['A'] * irr1['A']
                
                # String of product irrep, e.g. '(A6-7, A7-8)' to denote (7,8) of A6xA7 ~ SU(7)xSU(8)
                prod_str = '(' + irr0['irr_ID'] + ', ' + irr1['irr_ID'] + ')'

                # Store in array
                irrep_pairs.append([n0, n1, δH, AA, prod_str])

        # Now to find all possible combinations of product irreps

        # edges_admissible contains data on the H discount, bibj=sum(AA) entry in Gram matrix
        # and how many of each irrep are needed
        edges_admissible = []

        if len(irrep_pairs) == 0:
            # No possible product irreps
            return

        # Make numpy array for easier slicing
        irrep_pairs = np.array(irrep_pairs, dtype='object')
        
        # Backtrack through product irrep multiplicities (n_prod)
        n_prod = np.zeros(len(irrep_pairs), dtype='int')
        ii = 0

        done = False
        while not done:
            # Get the total number of irreps needed, δH and bibj info
            n0      = n_prod @ irrep_pairs[:, 0]
            n1      = n_prod @ irrep_pairs[:, 1]
            δH_prod = n_prod @ irrep_pairs[:, 2]
            bibj    = n_prod @ irrep_pairs[:, 3]

            # Check that there are enough individual irreps to "merge"
            if np.all(self.irreps['n'] >= n0) and np.all(other.irreps['n'] >= n1):

                # The 2x2 submatrix must have n_pos ≤ 1. This happens exactly when either
                # the trace or determinant is non-positive
                if self.bibi + other.bibi <= 0 or self.bibi*other.bibi - bibj**2 <= 0:
                        
                    # Construct irrep data (array of multiplicities and IDs)
                    irrep_data = [(n, string) for n, string in zip(n_prod, irrep_pairs[:, 4]) if n > 0]
                    irrep_data = np.array(irrep_data, dtype=dt_prod_irrep)

                    # Store in array as an admissible edge
                    edges_admissible.append([δH_prod, bibj, n0, n1, irrep_data])

                # Increment
                ii = 0
                n_prod[0] += 1

            elif ii == len(irrep_pairs) - 1:
                # End of backtracking
                done = True

            else:
                # Roll-over
                ii += 1
                n_prod[ii] += 1
                n_prod[:ii] = 0

        # For each edge, store both orientations: one from self → other and another
        # of the reversed orientation from other → self
        for edge_data in edges_admissible:

            # Only store nontrivial edges (check if δH=0)
            if edge_data[0] == 0:
                continue

            # Create the edge object and get the edge with reversed orientation
            edge = Edge(self, other, *edge_data)
            edge_reversed = edge.get_reverse()

            # Store based on the type of the vertices
            if other.type == 'A':
                self.edges_nontrivial_A.append(edge)
            else:
                self.edges_nontrivial_B.append(edge)

            # Store also at other as long as it's not an unoriented self-edge
            if (other is not self) or (edge is not edge_reversed):
                if self.type == 'A':
                    other.edges_nontrivial_A.append(edge_reversed)
                else:
                    other.edges_nontrivial_B.append(edge_reversed)

    def get_simple_clique(self, graph):
        """Returns the 1-clique consisting of only this vertex."""

        # Create a new clique object (setting branching mode based on vertex type)
        clique = IrreducibleClique(graph, self.type)

        # Add in the vertex (with no edges)
        clique.add_vertex(self, [])

        return clique
