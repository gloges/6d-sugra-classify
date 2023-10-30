import numpy as np


class ConstraintSolver():

    def __init__(self, num_irreps):
        """Facilitates the construction of solutions to the B-constraint for a simple group with `num_irreps` irreps.
        
        Data structure for organizing (B,H,n) tuples.
            - soln_B contains a list of B values:
                soln_B = [B1, B2, B3, ...]
            - soln_H is a list where each element corresponds
                to a value of B in soln_B and contains a list
                of H values for this value of B:
                    soln_H = [
                        [H11, H12, H13, ...]    ← w/ B1
                        [H21, H22, H23, ...]    ← w/ B2
                        [H31, H32, H33, ...]    ← w/ B3
                                . . .
                    ]
            - soln_n similarly contains the corresponding irrep
                multiplicities: the jth column of the ith row
                contains the list of ns which give B(i) and H(i,j).
        
        The advantage of this structure is that B/H bounds can
        be applied row-by-row rather than scanning through the
        full list of tuples.

        Args:
            num_irreps (int): The number of irreps for a simple group.

        """


        # Initialize to only the empty configuration with B=H=0 and no irreps
        self.soln_B = [0]
        self.soln_H = [np.zeros(1, dtype='int')]
        self.soln_n = [np.zeros([1, num_irreps], dtype='int')]

    def add_irrep(self, index, irrep, B_ub_min, H_ub):
        """Add in irrep at position `index`, subject to the upper bounds."""

        # Arrays for new solutions including the new irrep
        soln_B_new = []
        soln_H_new = []
        soln_n_new = []

        # Max number of new irrep that can be added
        n_max = max(H_ub) // irrep['H']

        for n_irrep in range(1, n_max + 1):

            # Loop through current solutions
            for B, Hs, ns in zip(self.soln_B, self.soln_H, self.soln_n):

                # Get new sum(nR*BR) when irrep ii is added n times
                B_new = B + n_irrep * irrep['B']

                if B_new < B_ub_min or B_new >= B_ub_min + len(H_ub):
                    # This value of B does not appear in the upper bound data,
                    # and so is not admissible (cannot be reached by future irreps)
                    continue

                # This value of B is admissible: get corresponding upper bound on H
                H_bound = H_ub[B_new - B_ub_min]

                # Get the subset of solutions which satisfy the bound when irrep ii is added
                subset = np.where(Hs + n_irrep * irrep['H'] <= H_bound)[0]
            
                if len(subset) > 0:
                    # There are new solutions
                    # Compute new H and irrep multiplicities and add to list to be returned

                    H_new = Hs[subset]
                    H_new += n_irrep * irrep['H']

                    n_new = ns[subset]
                    n_new[:, index] += n_irrep

                    soln_B_new.append(B_new)
                    soln_H_new.append(H_new)
                    soln_n_new.append(n_new)

        # Add in new solutions to data structure
        for B, Hs, ns in zip(soln_B_new, soln_H_new, soln_n_new):
            ii, todo = self.get_B_index(B)

            if todo == 'append':
                self.soln_B.append(B)
                self.soln_H.append(Hs)
                self.soln_n.append(ns)
            
            elif todo == 'insert':
                self.soln_B.insert(ii, B)
                self.soln_H.insert(ii, Hs)
                self.soln_n.insert(ii, ns)

            elif todo == 'merge':
                self.soln_H[ii] = np.append(self.soln_H[ii], Hs)
                self.soln_n[ii] = np.append(self.soln_n[ii], ns, axis=0)

    def get_B_index(self, B):
        """Returns information about where `B` appears in soln_B.

        Args:
            B (int): The value for B being searched for.

        Returns:
            i (int): Index where `B` appears / should appear.
            info (str): Information about whether `B` appears or where `B` should be inserted.
                            'append': `B` is larger than all values in soln_B
                            'merge' : `B` already appears at index i
                            'insert': `B` should be inserted at index i

        """

        # Use binary search to find where B should appear in array
        ii_low = 0
        ii_hgh = len(self.soln_B) - 1
        ii_mid = (ii_low + ii_hgh) // 2

        B_low = self.soln_B[ii_low]
        B_mid = self.soln_B[ii_mid]
        B_hgh = self.soln_B[ii_hgh]

        # Check edge cases
        if   B >  B_hgh: return ii_hgh, 'append'
        elif B == B_hgh: return ii_hgh, 'merge'
        elif B <  B_low: return ii_low, 'insert'
        elif B == B_low: return ii_low, 'merge'

        # Binary search until position for B is found
        while ii_hgh > ii_low + 1:

            if B_mid > B:
                ii_hgh = ii_mid
                B_hgh = B_mid
            else:
                ii_low = ii_mid
                B_low = B_mid
            
            ii_mid = (ii_low + ii_hgh) // 2
            B_mid = self.soln_B[ii_mid]

        # Check if B already exists
        if B == B_low:
            return ii_low, 'merge'
        else:
            return ii_hgh, 'insert'

    def remove_solutions(self, B_ub_min, H_ub):
        """Remove solutions which violate the upper bounds."""

        to_remove = []

        # Loop through current solutions
        for ii, [B, Hs, ns] in enumerate(zip(self.soln_B, self.soln_H, self.soln_n)):

            # Find this value of B in the upper bound data
            if B < B_ub_min or B >= B_ub_min + len(H_ub):
                # This value of B does not appear in the upper bound data,
                # and so is not admissible (cannot be reached by future irreps)
                to_remove.append(ii)

            else:
                # This value of B is admissible: get corresponding upper bound on H
                H_bound = H_ub[B - B_ub_min]

                # Find subset of solutions which satisfy the bound
                subset = np.where(Hs <= H_bound)[0]

                if len(subset) == 0:
                    # No solutions remain with this value of B: mark to remove
                    to_remove.append(ii)
                else:
                    # Restrict to solutions which satisfy the bound
                    self.soln_H[ii] = Hs[subset]
                    self.soln_n[ii] = ns[subset]

        # Loop through in reverse order to delete rows marked for removal
        for jj in to_remove[::-1]:
            self.soln_B.pop(jj)
            self.soln_H.pop(jj)
            self.soln_n.pop(jj)
