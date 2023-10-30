import numpy as np
from hashlib import sha1


# (d)ata(t)ype for storing product irreps: multiplicity and irrep ID
dt_prod_irrep = np.dtype([('n', 'int'), ('irr_ID', 'str', 255)])


class Edge():

    def __init__(self, v_tail, v_head, δH, bibj, irreps_tail, irreps_head, irreps):
        """Initializes a nontrivial edge from vertex `v_tail` to vertex `v_head`.

        Args:
            v_tail (Vertex): Vertex at the tail of the directed edge.
            v_head (Vertex): Vertex at the head of the directed edge.
            δH (int): Decrease to the number of hypers (δH ≤ 0).
            bibj (int): Off-diagonal Gram matrix entry.
            irreps_tail (array): Integer array giving the number of irreps used at v_tail.
            irreps_head (array): Integer array giving the number of irreps used at v_head.
            irreps (array): Array of (n, irrep_ID) pairs.

        """

        # Directed edge: v_tail → v_head
        self.tail = v_tail
        self.head = v_head

        self.δH               = δH              # H discount (δH <= 0)
        self.bibj             = bibj            # Off-diagonal Gram matrix entry
        self.irreps_tail      = irreps_tail     # Array of total irreps needed for v_tail
        self.irreps_head      = irreps_head     # Array of total irreps needed for v_head
        self.irreps           = irreps          # Array of (nR, irrep_string) pairs

        # A pointer to the edge of reversed orientation
        self.reverse = None

        # A string containing the bi-charged hypers (e.g. used in display())
        self.hypers_string = ' + '.join([str(n) + ' x ' + irr_ID for n, irr_ID in self.irreps])

        # Use a hash to create a unique ID (group ID plus 8 hex digits)
        # based on a string summarizing gauge groups and hypers
        tail_ID = self.tail.ID
        head_ID = self.head.ID
        tail_group = self.tail.ID[4:7]
        head_group = self.head.ID[4:7]

        hashdata = 'group = ' + tail_ID + 'x' + head_ID
        hashdata += '; hypers = ' + self.hypers_string
        hexhash = sha1(str.encode(hashdata)).hexdigest()[-8:]
        
        δHtext  = str(-self.δH).rjust(3, '0')

        self.ID = tail_group + 'x' + head_group + '-' + δHtext + '-' + hexhash

    def get_reverse(self):
        """Returns an edge with identical data but reversed orientation.
        If a self-edge is its own reverse then a new edge object is not created.

        Returns:
            Edge: Edge of reversed orientation.
        """

        # Create the irrep data for the reversed orientation
        # i.e. the product irrep (R,S) is replaced by (S,R)
        irreps_reversed = [(n, '(' + ', '.join(irr_ID[1:-1].split(', ')[::-1]) + ')')
                            for n, irr_ID in self.irreps]
        irreps_reversed = np.asarray(irreps_reversed, dtype=dt_prod_irrep)
        
        # If this is a self-edge, check if it is equal to its reverse
        if self.head is self.tail:
            symmetric = True

            # Sort both arrays
            irreps_sorted = np.sort(self.irreps, axis=0)
            irreps_reversed_sorted = np.sort(irreps_reversed, axis=0)

            # Loop through, checking if everything matches
            for aa, bb in zip(irreps_sorted, irreps_reversed_sorted):
                if aa[0] != bb[0] or aa[1] != bb[1]:
                    symmetric = False
                    break

            if symmetric:
                # Store a pointer to itself and return
                self.reverse = self
                return self

        # Otherwise, create the reversed edge
        edge_reversed = Edge(self.head, self.tail, self.δH, self.bibj,
                                self.irreps_head, self.irreps_tail, irreps_reversed)

        # Store references to reversed edges and return
        self.reverse = edge_reversed
        edge_reversed.reverse = self
        
        return edge_reversed

    def display(self):
        """Displays information about the nontrivial edge."""

        print(f'  {self.tail.ID} → {self.head.ID}')
        print('    ΔH =', self.δH)
        print('  bibj =', self.bibj)
        print('     ' + self.hypers_string)
        print()
