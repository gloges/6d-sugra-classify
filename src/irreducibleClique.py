import numpy as np
from itertools import product
from hashlib import sha1

from gramMatrix import signature, get_kernel_basis, get_T_min


class IrreducibleClique():

    def __init__(self, graph, branching_mode='any'):
        """Initializes an irreducible clique of `graph` with no vertices.

        Args:
            graph (Graph): Parent graph for this irreducible clique.

        """

        self.graph = graph
        self.Δ = 0

        # Arrays for vertex and edge indices
        self.vertices = []
        self.edges_nontrivial = []
        self.k = 0

        # Irreps used/available - keep a tally for each vertex
        self.irreps_used = []
        self.irreps_avbl = []

        # Gram matrix data
        self.gram_bibj = np.empty([0, 0], dtype='int')
        self.gram_b0bi = np.empty([0],    dtype='int')

        self.n_pos_bibj = 0
        self.n_neg_bibj = 0
        self.n_0_bibj   = 0

        # A useful quantity
        self.Δ_28n = self.Δ + 28*self.n_neg_bibj

        # Minimal T to be admissible
        self.T_min = 0

        # Numbers of types A and B vertices
        self.num_AB = [0, 0]

        # Active vertex indices
        self.active_vertices = []

        # Sets of bi which must be proportional
        # For each set the number of non-trivial edges to new vertices cannot be exactly one
        self.constrained_sets = []

        # 'any', 'A' or 'B'
        self.branching_mode = branching_mode

    def __lt__(self, other):

        if self.k != other.k:
            return self.k < other.k

        if self.Δ_28n != other.Δ_28n:
            return self.Δ_28n < other.Δ_28n
        
        for vertex_1, vertex_2 in zip(self.vertices, other.vertices):
            if vertex_1 is not vertex_2:
                return vertex_1 < vertex_2

        if len(self.edges_nontrivial) != len(other.edges_nontrivial):
            return len(self.edges_nontrivial) < len(other.edges_nontrivial)

        triples_1 = np.array([(jj-ii, ii, edge.ID) for ii, jj, edge in self.edges_nontrivial], dtype='object')
        triples_2 = np.array([(jj-ii, ii, edge.ID) for ii, jj, edge in other.edges_nontrivial], dtype='object')
        triples_1 = [aa for bb in triples_1.T for aa in bb]
        triples_2 = [aa for bb in triples_2.T for aa in bb]

        for aa, bb in zip(triples_1, triples_2):
            if aa != bb:
                return aa < bb

        return False
    
    def __eq__(self, other):

        if self.k != other.k:
            return False
        
        if self.Δ_28n != other.Δ_28n:
            return False
        
        for vertex_1, vertex_2 in zip(self.vertices, other.vertices):
            if vertex_1 is not vertex_2:
                return False

        if len(self.edges_nontrivial) != len(other.edges_nontrivial):
            return len(self.edges_nontrivial) < len(other.edges_nontrivial)

        triples_1 = np.array([(jj-ii, ii, edge.ID) for ii, jj, edge in self.edges_nontrivial], dtype='object')
        triples_2 = np.array([(jj-ii, ii, edge.ID) for ii, jj, edge in other.edges_nontrivial], dtype='object')
        triples_1 = [aa for bb in triples_1.T for aa in bb]
        triples_2 = [aa for bb in triples_2.T for aa in bb]

        for aa, bb in zip(triples_1, triples_2):
            if aa != bb:
                return False
        
        return True

    def display(self):
        """Displays detailed information about the state of the clique."""

        if self.k == 0:
            print('Empty clique')
            print('\n')
            return

        print(f'AB=({self.num_AB[0]},{self.num_AB[1]}) {self.k}-clique with...')
        print('      Δ =', self.Δ)
        print('  Δ+28n =', self.Δ_28n)
        print('  T_min =', self.T_min)

        values_nonzero = np.unique(self.gram_bibj)
        values_nonzero = values_nonzero[values_nonzero != 0]
        maxdig = 1
        if len(values_nonzero) > 0:
            maxdig += int(np.max([np.log10(np.abs(value)) + int(value < 0) for value in values_nonzero]))
        
        print('\n  b0.bi =  [' + ' '.join([str(xx).rjust(maxdig) for xx in self.gram_b0bi]) + ']')

        print('  bi.bj = [[', end='')

        for ii, row in enumerate(self.gram_bibj):
            if ii != 0:
                print('\n' + 11 * ' ' + '[', end='')
            strings = []
            for jj, xx in enumerate(row):
                if xx == 0 and jj != ii:
                    strings.append(maxdig*' ')
                else:
                    strings.append(str(xx).rjust(maxdig)) 
            print(' '.join(strings), end=']')
        print(']')

        print(f'  sign(bi.bj) = ({self.n_pos_bibj}, {self.n_neg_bibj}, {self.n_0_bibj})')

        print('\n  Vertices:')
        for ii, vertex in enumerate(self.vertices):
            print('   {:>6} = {:17}'.format('v_' + str(ii), vertex.ID), end=4*' ')
            print('  Δ = {:4}'.format(vertex.Δ), end=4*' ')
            print('  hypers =', vertex.irreps_string)

        print('\n  Non-trivial edges:')
        for ii, jj, edge in self.edges_nontrivial:
            edge_string = f'      {ii:2} → {jj:2} :    ΔH = {edge.δH:4}    hypers = '
            edge_string += ' + '.join([str(n) + ' x ' + string for n, string in edge.irreps])
            print(edge_string)

        print('\n  Irreps used / available / activity:')
        maxnum = max([len(aa) for aa in self.irreps_used])
        for ii, [used, avbl] in enumerate(zip(self.irreps_used, self.irreps_avbl)):
            print('   {:>6} :  '.format('v_' + str(ii)), end='')
            print('[' + ' '.join([str(xx).rjust(2) for xx in used]) + ']', end='')
            print((3*(maxnum-len(used)) + 2)*' ', end='')
            print('[' + ' '.join([str(xx).rjust(2) for xx in avbl]) + ']', end='')
            print((3*(maxnum-len(avbl)) + 2)*' ', end='')
            if ii in self.active_vertices:
                print('  active', end='')
            print()

        print('\n')

    def update_T_min(self):
        self.T_min = get_T_min(self.gram_b0bi, self.gram_bibj, self.T_min, self.graph.T_min_max)

    def update_constrained_sets(self):
        """Updates sets of vectors `bi` which much remain proportional when `n(+) = 1`."""

        # Determine sets of vertices which have bi = bj and thus must either
        # both have nontrivial or both have trivial edges to all other vertices

        # If n_pos = 0, then linearly dependent rows in bi.bJ need not
        # correspond to linearly dependent bi
        if self.n_pos_bibj == 0:
            self.constrained_sets = []

        else:
            # Build bi.bJ
            gram_aug = np.block([[self.gram_b0bi], [self.gram_bibj]]).T

            # Find linearly-dependent rows
            kernel_basis = get_kernel_basis(gram_aug)

            # Save the indices of vertices which appear in each linearly-dependent set
            self.constrained_sets = [list(np.where(row != 0)[0]) for row in kernel_basis]

    def get_nontrivial_adjacency(self):
        """Returns adjacency matrix restricted to nontrivial edges."""

        adj = np.zeros([self.k, self.k], dtype='int')
        
        for ii, jj, edge in self.edges_nontrivial:
            adj[ii,jj] = 1
            adj[jj,ii] = 1

        return adj

    def clone(self):
        """Creates a copy of the clique."""

        clone = IrreducibleClique(self.graph)

        clone.k     = self.k
        clone.Δ     = self.Δ
        clone.Δ_28n = self.Δ_28n

        clone.vertices         = self.vertices.copy()
        clone.active_vertices  = self.active_vertices.copy()
        clone.edges_nontrivial = self.edges_nontrivial.copy()

        clone.gram_bibj  = self.gram_bibj.copy()
        clone.gram_b0bi  = self.gram_b0bi.copy()
        clone.n_pos_bibj = self.n_pos_bibj
        clone.n_neg_bibj = self.n_neg_bibj
        clone.n_0_bibj   = self.n_0_bibj

        clone.irreps_used = [used.copy() for used in self.irreps_used]
        clone.irreps_avbl = [avbl.copy() for avbl in self.irreps_avbl]

        clone.num_AB = self.num_AB.copy()
        clone.T_min   = self.T_min

        clone.maximum_vertex = self.maximum_vertex
        clone.branching_mode = self.branching_mode

        return clone

    def sort(self):
        """Brings the vertices into a standard order.
        
        The standard order has all type-A vertices before all type-B vertices
        and prioritizes having `gram=bi.bj` as diagonal as possible, traversing
        the graph formed by the non-trivial edges with backtracking.
        
        """

        if self.k <= 1:
            # 0- and 1-cliques: nothing to do!
            return

        # First get adjacency matrix and identify type-A vertices
        adjacency_matrix = self.get_nontrivial_adjacency()
        typeA = [vertex.type == 'A' for vertex in self.vertices]

        # Find roots for tree
        if any(typeA):
            # If type-A present start there
            roots = np.where(typeA)[0]
        else:
            # Otherwise find vertices of lowest degree (often 1)
            vertex_degrees = np.sum(adjacency_matrix, axis=1)
            roots = np.where(vertex_degrees == min(vertex_degrees))[0]

        # Save current order and reference data for comparison with candidate reorderings
        current_order = list(range(self.k))
        data_ref = self.reordered_edge_data(current_order)
        data_ref = [self.vertices, *data_ref]
        data_ref = [aa for bb in data_ref for aa in bb]

        # Construct recursively all potential paths through the graph of nontrivial edges
        # Paths are grown in chunks until a leaf is hit. At each step only the paths
        # which have the longest chunk are kept.

        paths = [[root] for root in roots]
        while len(paths[0]) < self.k:

            # Grow paths by another chunk
            paths_longer = []
            for path in paths:
                paths_longer.extend(grow_path(path, adjacency_matrix, typeA))
            
            # Restrict to subset for which the path is longest (because the new chunk is longest)
            longest = max([len(path) for path in paths_longer])
            paths_longer = [path for path in paths_longer if len(path) == longest]

            # Remove paths which are related by symmetries of the clique
            ii = 0
            while ii < len(paths_longer):
                # Walk through array backwards, deleting if identical
                for jj in range(len(paths_longer)-1, ii, -1):
                    # Get permutation which takes you from one path to the other
                    order = list(range(self.k))
                    for aa, bb in zip(paths_longer[ii], paths_longer[jj]):
                        order[aa], order[bb] = bb, aa

                    if len(order) != len(np.unique(order)):
                        # Not comparable: keep both
                        continue

                    # Gather data
                    data = self.reordered_edge_data(order)
                    data = [self.vertices[order], *data]
                    data = [aa for bb in data for aa in bb]

                    # Check if identical
                    identical = True
                    for aa, bb in zip(data, data_ref):
                        if aa != bb:
                            identical = False
                            break

                    if identical:
                        # Remove duplicate
                        paths_longer.pop(jj)

                ii += 1
            

            # If paths are complete, we're done!
            if len(paths_longer[0]) == self.k:
                paths = paths_longer
                break

            # Otherwise, seed each path with one more node
            paths = []
            for path in paths_longer:
                # Walk backwards up path until node with unused neighbor is found
                ii = len(path)
                neighbors = []
                while len(neighbors) == 0:
                    ii -= 1
                    neighbors = np.where(adjacency_matrix[path[ii]] > 0)[0]
                    neighbors = [neighbor for neighbor in neighbors if neighbor not in path]
                
                # Grow path by each such neighbor
                paths.extend([[*path, neighbor] for neighbor in neighbors])
        
        # paths now contains a sequence of paths, all of which have the same
        # (largest) chunk sizes before backtrackings. Now to find the best
        # new ordering based on edge data (vertex proximity and IDs)

        # Keep track of optimal edge data and corresponding order
        data_best = None
        path_best = None
        for path in paths:
            # Get data which will determine which ordering of vertices is best
            data = self.reordered_edge_data(path)
            data = [(jj-ii, ii, edge.ID) for ii, jj, edge in data]
            data = np.array(data, dtype='object')
            data = [aa for bb in data.T for aa in bb]
            data = [*data, *[self.vertices[ii].ID for ii in path]]

            # data now contains, in order, all differences jj-ii, then all indices ii,
            # then all edge IDs, then all vertex IDs. The best path is the one that
            # is lexicographically first.

            # Determine if current path through vertices is best so far
            update_best = False

            if path_best is None:
                update_best = True
            else:
                # Check against current best
                for aa, bb in zip(data, data_best):
                    if aa < bb:
                        update_best = True
                    if aa != bb:
                        break

            if update_best:
                data_best = data
                path_best = path

        # Sort based on the best path through the vertices
        self.sort_by_order(path_best)

    def reordered_edge_data(self, order):
        """Returns the cliques edge data if vertices were to be reordered according to `order`."""

        # Sort nontrivial edge data according to order
        edges_nontrivial_new = []
        for ii, jj, edge in self.edges_nontrivial:
            # Find where vertices ii and jj would end up
            ii_new = np.where(np.array(order) == ii)[0][0]
            jj_new = np.where(np.array(order) == jj)[0][0]

            # Reorient edge if needed
            if ii_new < jj_new:
                edges_nontrivial_new.append((ii_new, jj_new, edge))
            else:
                edges_nontrivial_new.append((jj_new, ii_new, edge.reverse))

        # Standard order for nontrivial edge data:
        triples = np.array([(jj-ii, ii, edge.ID) for ii, jj, edge in edges_nontrivial_new], dtype='object')
        edge_order = np.lexsort(triples.T[::-1])

        return [edges_nontrivial_new[ii] for ii in edge_order]

    def sort_by_order(self, order):
        """Sorts the cliques vertices according to `order`."""

        # Sort vertices
        self.vertices = self.vertices[order]

        # Sort nontrivial edges
        self.edges_nontrivial = self.reordered_edge_data(order)

        # Sort irreps used / available
        self.irreps_used = [self.irreps_used[ii] for ii in order]
        self.irreps_avbl = [self.irreps_avbl[ii] for ii in order]

        # Reorder Gram matrix data
        self.gram_bibj = self.gram_bibj[order][:, order]
        self.gram_b0bi = self.gram_b0bi[order]

        # Constrained sets
        self.constrained_sets = [sorted([np.where(order == ii)[0][0] for ii in constrained_set])
                                    for constrained_set in self.constrained_sets]
        
        self.active_vertices = [np.where(np.array(order) == ii)[0][0] for ii in self.active_vertices]

    def get_maximum_edge_discount(self):
        """Returns sum(sum(n_avbl(v)*H : n_avbl >= Hmin) : active vertices)."""

        nH_sum = 0

        for ii in self.active_vertices:
            vertex, vertex_irreps_avbl = self.vertices[ii], self.irreps_avbl[ii]
            for H, n_avbl in zip(vertex.irreps['H'], vertex_irreps_avbl):
                if n_avbl >= self.graph.min_irrep_H:
                    nH_sum += n_avbl * H

        return nH_sum

    def get_edge_candidates(self, ii):
        """Returns a list of viable edges for the `ii`th vertex to type-A/B vertices according to the branching mode."""

        vertex = self.vertices[ii]
        vertex_irreps_avbl = self.irreps_avbl[ii]

        edge_candidates = [None]

        if ii in self.active_vertices:

            if self.branching_mode in ['A', 'any']:
                edge_candidates.extend([edge for edge in vertex.edges_nontrivial_A
                                        if all(edge.irreps_tail <= vertex_irreps_avbl)])

            if self.branching_mode in ['B', 'any']:
                edge_candidates.extend([edge for edge in vertex.edges_nontrivial_B
                                        if all(edge.irreps_tail <= vertex_irreps_avbl)])

        return edge_candidates

    def branch_by_vertex(self):
        """Returns a list of irreducible cliques with one additional vertex.

        Vertices from the neighborhood of the clique (with type dictated by
        the branching mode) are considered along with all possible choices
        of edges to pre-existing vertices. Only admissible irreducible cliques
        are returned.

        Returns:
            list: Admissible irreducible (k+1)-cliques
        """


        cliques_branched = []

        edge_candidates = [self.get_edge_candidates(ii) for ii in range(self.k)]

        # Identify neighborhood of vertices connected to the clique by one or more viable nontrivial edges
        neighborhood = [[edge.head for edge in edges if edge is not None] for edges in edge_candidates]
        neighborhood = np.unique([aa for bb in neighborhood for aa in bb])

        # Keep track of which active vertices successfully participate in a branching
        successfully_branched = np.full(self.k, False)

        # Branch by each neighbor in turn
        for neighbor in neighborhood:

            # Collect edges from each pre-existing vertex to this neighbor
            edges_to_neighbor = [[edge for edge in edges if edge is None or edge.head is neighbor]
                                    for edges in edge_candidates]
            
            # Check all combinations of edges which respect irrep usage at neighbor
            edge_combos = self.get_nontrivial_edge_combos(neighbor, edges_to_neighbor)
            for edges_new in edge_combos:

                clique_new = self.clone()
                clique_new.add_vertex(neighbor, edges_new)

                if np.isfinite(clique_new.T_min) and clique_new.T_min <= self.graph.T_min_max:
                    cliques_branched.append(clique_new)

                    for ii, edge in enumerate(edges_new):
                        if edge is not None:
                            successfully_branched[ii] = True

        # Update new cliques' active vertices and *then* sort
        for clique_new in cliques_branched:
            active_vertices_new = clique_new.active_vertices.copy()
            
            for ii in clique_new.active_vertices:
                if ii < clique_new.k - 1 and not successfully_branched[ii]:
                    # A pre-existing vertex that *never* successfully branched
                    # This overrides all else
                    active_vertices_new.remove(ii)
                    continue

                if len(clique_new.irreps_avbl[ii]) == 0 or max(clique_new.irreps_avbl[ii]) < clique_new.graph.min_irrep_H:
                    active_vertices_new.remove(ii)

            clique_new.active_vertices = active_vertices_new
            clique_new.sort()

        cliques_branched = unique_cliques(cliques_branched)

        return cliques_branched

    def branch_by_B_clique(self, clique_B):
        """Returns a list of irreducible cliques formed by joining to `clique_B` by one or more nontrivial edge.

        This clique and `clique_B` are joined via all possible choices for non-trivial edges from
        type-A vertices to type-B vertices.

        Args:
            clique_B (IrreducibleClique): A type-B irreducible clique

        Returns:
            list: Irreducible cliques
        """

        # Get all edge combos which respect the irrep usage of both cliques
        edge_combos = self.get_nontrivial_edge_combos_to_B_clique(clique_B)

        # For each combo, glue together and check for admissibility.
        cliques_joined = []
        for edge_combo in edge_combos:
            clq_joined = self.clone()
            clq_joined.add_B_clique(clique_B, edge_combo)
            clq_joined.sort()

            if np.isfinite(clq_joined.T_min) and clq_joined.T_min <= self.graph.T_min_max:
                cliques_joined.append(clq_joined)

        cliques_joined = unique_cliques(cliques_joined)

        return cliques_joined

    def get_nontrivial_edge_combos(self, vertex, edges_to_vertex):
        """Returns combinations of nontrivial edges from this clique to `vertex`, given a list of viable edges."""

        # Recursively find combos
        # (along the way use the constrained sets)

        # Start with edge combos for *zero* vertices and one-by-one add in edges to each vertex
        edge_combos = [[]]
        for edge_cands in edges_to_vertex:

            # Add to each combo of edges to (n) vertices all possible edges for the next vertex
            if vertex.type == 'A':
                # type-A vertex is being added (implicitly to a type-A clique): no trivial edges are allowed
                edge_combos_next = [[[*edge_combo, edge] for edge in edge_cands if edge is not None]
                                    for edge_combo in edge_combos]
            else:
                # type-B vertex is being added: trivial edges are allowed
                edge_combos_next = [[[*edge_combo, edge] for edge in edge_cands] for edge_combo in edge_combos]

            edge_combos_next = [aa for bb in edge_combos_next for aa in bb]

            # Prepare to fill with combos to (n+1)-vertices
            edge_combos = []
            for edge_combo in edge_combos_next:

                # Check irrep usage
                irreps_used = np.zeros(len(vertex.irreps), dtype='int')
                for edge in edge_combo:
                    if edge is not None:
                        irreps_used += edge.irreps_head
                if any(irreps_used > vertex.irreps['n']):
                    continue

                # Check constrained sets
                constraints_okay = True
                for constrained_set in self.constrained_sets:

                    # Only should check if all vertices of the constrained set have been added
                    if max(constrained_set) < len(edge_combo):
                        # There cannot be exactly one non-trivial edge to the vertices
                        # of a constrained set
                        num_nontrivial = 0
                        for ii in constrained_set:
                            if edge_combo[ii] is not None:
                                num_nontrivial += 1

                        if num_nontrivial == 1:
                            constraints_okay = False

                if not constraints_okay:
                    continue

                edge_combos.append(edge_combo)

        # Remove the all-trivial edge combo if present (it will be first)
        if len(edge_combos) > 0 and all([edge is None for edge in edge_combos[0]]):
            edge_combos.pop(0)

        return edge_combos

    def get_nontrivial_edge_combos_to_B_clique(self, clique_B):
        """Returns all viable combinations of edges between this clique and `clique_B`."""

        # Identify combos of edges from self (type-A or type-AB) to the B-clique
        # Only edges from type-A vertices of self to vertices of clique_B

        # First identify viable edges
        edge_candidates = []
        for ii in range(self.k):
            for edge in self.get_edge_candidates(ii):
                if edge is not None:
                    for jj in clique_B.active_vertices:
                        if edge.head is clique_B.vertices[jj] and all(edge.irreps_head <= clique_B.irreps_avbl[jj]):
                            edge_candidates.append([ii, jj, edge])

        # Find combos respecting irrep usage. Try adding each edge to candidate combinations one by one
        edge_combos = [np.full([self.k, clique_B.k], None)]
        for ii, jj, edge in edge_candidates:
            edge_combos_new = []

            for edge_combo in edge_combos:
                if edge_combo[ii,jj] is not None:
                    # An edge between these vertices already exists in this edge combo
                    continue

                # Create a new edge combination with this nontrivial edge
                edge_combo_new = edge_combo.copy()
                edge_combo_new[ii,jj] = edge

                # Check irrep usage
                irreps_used_A = [0*v_avbl for v_avbl in self.irreps_avbl]
                irreps_used_B = [0*v_avbl for v_avbl in clique_B.irreps_avbl]
                for kk, ll in product(range(self.k), range(clique_B.k)):
                    if edge_combo_new[kk,ll] is not None:
                        irreps_used_A[kk] += edge_combo_new[kk,ll].irreps_tail
                        irreps_used_B[ll] += edge_combo_new[kk,ll].irreps_head

                irreps_okay = True
                for n_used_A, n_avbl in zip(irreps_used_A, self.irreps_avbl):
                    if any(n_used_A > n_avbl):
                        irreps_okay = False
                        break
                for n_used_B, n_avbl in zip(irreps_used_B, clique_B.irreps_avbl):
                    if any(n_used_B > n_avbl):
                        irreps_okay = False
                        break

                if irreps_okay:
                    edge_combos_new.append(edge_combo_new)

            edge_combos.extend(edge_combos_new)

        # Remove combo which is all trivial edges
        edge_combos = edge_combos[1:]

        return edge_combos

    def add_vertex(self, vertex, edges_new):
        """Adds a vertex to this clique connected to the pre-existing vertices via the edges in `edges_new`."""

        # Add to vertices and update k, Δ
        self.vertices = np.append(self.vertices, vertex)
        self.k += 1
        self.Δ += vertex.Δ

        # Add in b0.bi and diagonal bi.bj element
        self.gram_bibj = np.pad(self.gram_bibj, (0, 1))
        self.gram_b0bi = np.append(self.gram_b0bi, 0)
        vertex.add_gram_entries(self, -1)

        # Prepare irrep usage/availability arrays
        self.irreps_used.append(0 * vertex.irreps['n'])
        self.irreps_avbl.append(1 * vertex.irreps['n'])

        # Update list of edges, off-diagonal bi.bj elements,
        # Δ and irrep usage/availability
        for ii, edge in enumerate(edges_new):

            # For trivial edges nothing needs to be done
            if edge is None:
                continue

            # Add to list of nontrivial edges
            self.edges_nontrivial.append((ii, self.k-1, edge))

            # Off-diagonal bi.bj elements
            self.gram_bibj[ii, -1] = edge.bibj
            self.gram_bibj[-1, ii] = edge.bibj

            # Change to number of hypers
            self.Δ += edge.δH

            # Irrep usage/availability
            self.irreps_used[ii] += edge.irreps_tail
            self.irreps_avbl[ii] -= edge.irreps_tail
            self.irreps_used[-1] += edge.irreps_head
            self.irreps_avbl[-1] -= edge.irreps_head

        # Update bi.bj signature and Δ+28n_neg
        self.n_pos_bibj, self.n_neg_bibj, self.n_0_bibj = signature(self.gram_bibj)
        self.Δ_28n = self.Δ + 28*self.n_neg_bibj

        # Update clique type
        if vertex.type == 'A':
            self.num_AB[0] += 1
        else:
            self.num_AB[1] += 1

        if self.num_AB[1] == 0:
            # If still a type-A clique, update maximum vertex
            self.maximum_vertex = vertex
        
        elif self.k == 1 and self.num_AB[0] == 0:
            # If this is a type-B simple clique, set maximum vertex
            self.maximum_vertex = vertex

        # Update Tmin, constrained sets and edge candidates
        if self.n_pos_bibj > 1:
            self.T_min = np.inf
        else:
            self.T_min = max(self.T_min, vertex.T_min)
            if self.k > 1:
                self.update_T_min()
        
        self.update_constrained_sets()

        self.active_vertices.append(self.k - 1)

        for ii in self.active_vertices[::-1]:
            if len(self.irreps_avbl[ii]) == 0 or max(self.irreps_avbl[ii]) < self.graph.min_irrep_H:
                self.active_vertices.remove(ii)

    def add_B_clique(self, clique_B, edges_new):
        """Joins `clique_B` to this clique, connecting vertices of the two pieces via edges in `edges_new`."""

        k_original = self.k

        # Add to vertices and update k, Δ
        self.vertices = np.append(self.vertices, clique_B.vertices)
        self.k += clique_B.k
        self.Δ += clique_B.Δ

        # Add in b0.bi and diagonal bi.bj element
        zeros = np.zeros([k_original, clique_B.k], dtype=int)
        self.gram_bibj = np.block([
            [self.gram_bibj, zeros             ],
            [zeros.T,        clique_B.gram_bibj]
        ])
        self.gram_b0bi = np.append(self.gram_b0bi, clique_B.gram_b0bi)

        for ii, jj, edge in clique_B.edges_nontrivial:
            self.edges_nontrivial.append((k_original + ii, k_original + jj, edge))

        # Prepare irrep usage/availability arrays
        self.irreps_used.extend([v_used.copy() for v_used in clique_B.irreps_used])
        self.irreps_avbl.extend([v_avbl.copy() for v_avbl in clique_B.irreps_avbl])

        # Update list of edges, off-diagonal bi.bj elements,
        # Δ and irrep usage/availability
        for ii, jj in product(range(k_original), range(clique_B.k)):
            edge = edges_new[ii,jj]

            # For trivial edges nothing needs to be done
            if edge is None:
                continue

            # Add to list of nontrivial edges
            self.edges_nontrivial.append((ii, k_original + jj, edge))

            # Off-diagonal bi.bj elements
            self.gram_bibj[ii, k_original + jj] = edge.bibj
            self.gram_bibj[k_original + jj, ii] = edge.bibj

            # Change to number of hypers
            self.Δ += edge.δH

            # Irrep usage/availability
            self.irreps_used[ii] += edge.irreps_tail
            self.irreps_avbl[ii] -= edge.irreps_tail

            self.irreps_used[k_original + jj] += edge.irreps_head
            self.irreps_avbl[k_original + jj] -= edge.irreps_head

        # Bring edge data to standard order
        self.edges_nontrivial = self.reordered_edge_data(list(range(self.k)))

        # Update bi.bj signature and Δ+28n_neg
        self.n_pos_bibj, self.n_neg_bibj, self.n_0_bibj = signature(self.gram_bibj)
        self.Δ_28n = self.Δ + 28*self.n_neg_bibj

        # Update clique type
        self.num_AB[0] += clique_B.num_AB[0]
        self.num_AB[1] += clique_B.num_AB[1]

        # Update Tmin, constrained sets and edge candidates
        if self.n_pos_bibj > 1:
            self.T_min = np.inf
        else:
            self.T_min = max(self.T_min, clique_B.T_min)
            if self.k > 1:
                self.update_T_min()
        
        self.update_constrained_sets()

        for ii in self.active_vertices[::-1]:
            if len(self.irreps_avbl[ii]) == 0 or max(self.irreps_avbl[ii]) < self.graph.min_irrep_H:
                self.active_vertices.remove(ii)

    def build_AB_cliques(self, cliques_B_n_pos_0, cliques_B_n_pos_restricted_0, threshold=300):
        """Returns all bounded, irreducible cliques formed by joining this clique to type-B cliques.

        This function should be applied only to type-A cliques.
        The list `cliques_B_n_pos_0` (`cliques_B_n_pos_restricted_0`) contains type-B cliques
        for which `n_pos_bibj` (`n_pos_bibj` when restricted to inactive vertices) is zero. Type-B cliques
        are joined to the type-A clique via (possibly multiple) nontrivial edges to form type-AB cliques.
        This is repeatedly done, pruning along the way, until all bounded cliques with this type-A subclique
        are produced.

        If bounded cliques are thus produced, this clique itself is included in the returned list
        if it does not satisfy the bound Δ+28n_neg + T_min <= 273 (and appears first).

        Args:
            cliques_B_n_pos_0 (list): List of type-B cliques which have n_pos_bibj = 0. Assumed sorted in order of increasing IrreducibleClique.f(0.5, False).
            cliques_B_n_pos_restricted_0 (list): List of type-B cliques which have n_pos = 0 when restricted to inactive vertices. Assumed sorted in order of increasing IrreducibleClique.f(0.5, False).
            threshold (int, optional): Threshold for pruning. Defaults to 300.

        Returns:
            list: List of irreducible type-AB cliques with Δ + 28*n_neg + T_min <= 273.
        """

        # type-AB cliques, collected by how many type-B irreducible cliques
        # have been glued on (always to the type-A vertices)
        cliques_AB = [[self]]

        while len(cliques_AB[-1]) > 0:
            cliques_new = []

            for clq in cliques_AB[-1]:

                # 
                max_value = threshold - clq.prune_function(improved=True)

                for clqB in cliques_B_n_pos_0:
                    if clqB.f(0.5, False) > max_value:
                        # Too large: abort!
                        break

                    # Join by all possible edges
                    cliques_new.extend([cc for cc in clq.branch_by_B_clique(clqB)
                                        if cc.prune_function(improved=True) <= threshold])

                # If n_pos_restricted is zero, can possibly join to type-B cliques
                # which have n_pos = 1 but n_pos_restricted = 0
                if clq.signature_gram_bibj_restricted(clq.active_vertices)[0] == 0:
                    for clqB in cliques_B_n_pos_restricted_0:
                        if clqB.f(0.5, False) > max_value:
                            # Too large: abort!
                            break

                        # Join by all possible edges
                        cliques_new.extend([cc for cc in clq.branch_by_B_clique(clqB)
                                            if cc.prune_function(improved=True) <= threshold])

            cliques_new = unique_cliques(cliques_new)
            cliques_AB.append(cliques_new)

            # Clean up previous list of cliques to only those which are bounded (don't remove self)
            if len(cliques_AB) > 2:
                cliques_AB[-2] = [clq for clq in cliques_AB[-2] if clq.Δ_28n + clq.T_min <= 273]

        # Flatten. The first clique in cliques_AB is always 'self'
        cliques_AB = [aa for bb in cliques_AB for aa in bb]

        if len(cliques_AB) == 1:
            # Only 'self'
            return []
        else:
            # More cliques than just 'self'. Only keep 'self' if it doesn't satisfy the bound.
            if self.Δ_28n + self.T_min <= 273:
                return cliques_AB[1:]
            else:
                return cliques_AB

    def signature_gram_bibj_restricted(self, to_remove):
        """Returns the signature of the Gram matrix bi.bj when some rows/columns removed."""

        gram_bibj_restricted = self.gram_bibj.copy()
        gram_bibj_restricted = np.delete(gram_bibj_restricted, to_remove, axis=0)
        gram_bibj_restricted = np.delete(gram_bibj_restricted, to_remove, axis=1)

        return signature(gram_bibj_restricted)

    def f(self, λ, restrict_n_neg):
        """Returns Δ+28n_neg - λ*maximum_edge_discount. `restrict_n_neg` removes active vertices from bi.bj."""

        if restrict_n_neg:
            n_neg = self.signature_gram_bibj_restricted(self.active_vertices)[1]
        else:
            n_neg = self.n_neg_bibj

        return self.Δ + 28*n_neg - λ*self.get_maximum_edge_discount()

    def prune_function(self, improved=False):
        """Returns the value of a function used to determine when to prune cliques.

        For improved=False, uses the most naive method. For improved=True, uses
        an improved method which is more stringent. The improvement requires having
        set the graph vertex_neg_values. TODO: See appendix

        """


        if not improved:
            # Naive method
            return self.f(1, True) + self.T_min
        
        else:
            # Improved method

            # Correction terms for each active vertex which can form edge(s)
            # to a type-B clique with Δ+28n - 0.5*sum(nH) < 0
            correction = 0

            for ii in self.active_vertices:
                pairs = []
                for edge in self.get_edge_candidates(ii):
                    if edge is not None:
                        neg_value = self.graph.get_neg_value(edge.head.index, edge.irreps_head)
                        if neg_value < 0:
                            pairs.append([edge.irreps_tail, neg_value])

                if len(pairs) > 0:
                    # Solve knapsack problem
                    item_weights = [aa for aa, bb in pairs]
                    item_values = [-bb for aa, bb in pairs]
                    max_weights = self.irreps_avbl[ii]

                    correction -= knapsack(item_weights, item_values, max_weights)

            return self.f(0.5, True) + self.T_min + correction

    def get_save_string(self):
        """Returns a tsv string representation of the clique.
        
        This string is human-readable, but is not sufficiently rich to reconstruct
        an `IrreducibleClique` object. See also `get_restorable_save_string()`.
        
        """

        # TODO: summary of data in doc-string

        # (t)ab-(s)eparated (v)alues:
        #   - Irreducible clique ID: clq_[numA]_[numB]_[npos]_[nneg]_[Δ]_[hash]
        #   - Array of vertex IDs
        #   - Δ
        #   - Δ+28n
        #   - Tmin
        #   - npos, nneg
        #   - bi.bj
        #   - b0.bi
        #   - Hypers

        data_string = '[' + ', '.join([vertex.ID for vertex in self.vertices]) + ']'
        data_string += '\tΔ = ' + str(self.Δ)
        data_string += '\tΔ+28n = ' + str(self.Δ_28n)

        data_string += '\tTmin = ' + str(self.T_min)
        data_string += '\tnpos,nneg = [' + str(self.n_pos_bibj) + ', ' + str(self.n_neg_bibj) + ']'

        data_string += '\tbi.bj = [' + ', '.join(['[' + ', '.join([str(xx) for xx in row]) + ']' for row in self.gram_bibj]) + ']'
        data_string += '\tb0.bi = [' + ', '.join([str(xx) for xx in self.gram_b0bi]) + ']'

        for ii, [vertex, vertex_avbl] in enumerate(zip(self.vertices, self.irreps_avbl)):
            avbl_data = [str(n_avbl) + ' x (' + irr_ID + ')'
                            for n_avbl, irr_ID in zip(vertex_avbl, vertex.irreps['irr_ID'])
                            if n_avbl > 0]

            if len(avbl_data) > 0:
                data_string += f'\t({ii+1}) : '
                data_string += ' + '.join(avbl_data)

        for ii, jj, edge in self.edges_nontrivial:
            data_string += f'\t({ii+1},{jj+1}) : '
            data_string += edge.hypers_string

        # Create a unique ID by hashing the data string so far
        data_hash = sha1(str.encode(data_string)).hexdigest()[-8:]

        prefix = f'clq-({self.num_AB[0]},{self.num_AB[1]})-({self.n_pos_bibj},{self.n_neg_bibj})-'
        if self.Δ < 0:
            prefix += 'n'
        prefix += f'{abs(self.Δ)}-' + data_hash

        data_string = prefix + '\t' + data_string

        return data_string

    def get_restorable_save_string(self):
        """Returns a tsv string representation of the clique.
        
        This string representation is sufficiently rich to reconstruct
        an `IrreducibleClique` object using `restore_from_tsv()`. See also `get_save_string()`.
        
        """

        data_string = '[' + ', '.join([str(vertex.index) for vertex in self.vertices]) + ']'
        data_string += '\tΔ = ' + str(self.Δ)
        data_string += '\tΔ+28n = ' + str(self.Δ_28n)

        data_string += '\tbi.bj = [' + ', '.join(['[' + ', '.join([str(xx) for xx in row]) + ']' for row in self.gram_bibj]) + ']'
        data_string += '\tb0.bi = [' + ', '.join([str(xx) for xx in self.gram_b0bi]) + ']'

        data_string += '\tT_min = ' + str(self.T_min)

        data_string += '\tedges = ['
        data_string += ', '.join([f'({ii}, {jj}, {edge.ID})' for ii, jj, edge in self.edges_nontrivial])
        data_string += ']'

        data_string += '\tbranching mode = ' + self.branching_mode

        data_string += '\tactive = ['
        data_string += ', '.join([str(aa) for aa in self.active_vertices])
        data_string += ']'

        prefix = f'clq-({self.num_AB[0]},{self.num_AB[1]})-({self.n_pos_bibj},{self.n_neg_bibj})-'
        if self.Δ < 0:
            prefix += 'n'
        prefix += f'{abs(self.Δ)}'

        data_string = prefix + '\t' + data_string

        return data_string

def restore_from_tsv(graph, data_string):
    """Returns an `IrreducibleClique` object reconstructed from `data_string` previously generated by `get_restorable_save_string()`."""

    data = data_string.split('\t')

    clique = IrreducibleClique(graph)
    clique.num_AB = [int(aa) for aa in data[0].split('(')[1].split(')')[0].split(',')]

    # Vertices
    vertex_indices  = [int(ii) for ii in data[1][1:-1].split(', ')]
    clique.vertices = graph.vertices[vertex_indices]
    clique.k        = len(clique.vertices)

    # Maximum vertex
    clique.maximum_vertex = clique.vertices[0]
    for vertex in clique.vertices:
        if vertex > clique.maximum_vertex:
            clique.maximum_vertex = vertex
        elif vertex.type == 'A' and clique.maximum_vertex.type == 'A' and vertex < clique.maximum_vertex:
            clique.maximum_vertex = vertex
    
    # Δ, Δ+28n
    clique.Δ = int(data[2][4:])
    clique.Δ_28n = int(data[3][8:])

    # Gram matrix data
    clique.gram_bibj = np.array([[int(entry) for entry in row.split(', ')] for row in data[4][10:-2].split('], [')])
    clique.gram_b0bi = np.array([int(entry) for entry in data[5][9:-1].split(', ')])
    clique.n_pos_bibj, clique.n_neg_bibj, clique.n_0_bibj = signature(clique.gram_bibj)
    
    # Tmin
    clique.T_min = int(data[6][8:])

    # Nontrivial edges
    edge_data = data[7].split('[(')
    if len(edge_data) == 1:
        edge_data = []
    else:
        edge_data = [trip.split(', ') for trip in edge_data[1].split(')]')[0].split('), (')]
    clique.edges_nontrivial = []
    for ii_str, jj_str, edgeID in edge_data:
        ii = int(ii_str)
        jj = int(jj_str)

        # Find edge object from vertex ii to vertex jj with this edgeID
        if clique.vertices[jj].type == 'A':
            edge_list = clique.vertices[ii].edges_nontrivial_A
        else:
            edge_list = clique.vertices[ii].edges_nontrivial_B

        for edge in edge_list:
            if edge.head is clique.vertices[jj] and edge.ID == edgeID:
                clique.edges_nontrivial.append((ii, jj, edge))
                break

    # Info for branching
    clique.branching_mode = data[8][17:]
    if len(data[9]) == 11:
        clique.active_vertices = []
    else:
        clique.active_vertices = [int(ii) for ii in data[9].split('[')[1].split(']')[0].split(', ')]

    # Build irreps used/avbl based on the above
    clique.irreps_used = [0 * vertex.irreps['n'] for vertex in clique.vertices]
    clique.irreps_avbl = [1 * vertex.irreps['n'] for vertex in clique.vertices]

    for ii, jj, edge in clique.edges_nontrivial:
        clique.irreps_used[ii] += edge.irreps_tail
        clique.irreps_used[jj] += edge.irreps_head
        
        clique.irreps_avbl[ii] -= edge.irreps_tail
        clique.irreps_avbl[jj] -= edge.irreps_head

    return clique

def unique_cliques(cliques):
    """Returns a list of sorted cliques and with duplicates removed."""

    # First sort the array: any duplicates will now be back-to-back
    cliques = np.sort(cliques)

    # Identify duplicates by comparing pairwise
    to_delete = []
    for ii in range(len(cliques)-1):
        if cliques[ii] == cliques[ii+1]:
            cliques[ii+1].active_vertices = list(np.intersect1d(cliques[ii].active_vertices,
                                                                cliques[ii+1].active_vertices))
            to_delete.append(ii)

    # Delete duplicates and return
    cliques = np.delete(cliques, to_delete)
    return cliques

def grow_path(path, adjacency_matrix, typeA):
    """For constructing paths through connected components recursively."""

    # Identify unused nodes connected to end of path
    whr = np.where(adjacency_matrix[path[-1]] == 1)[0]
    whr = [ii for ii in whr if ii not in path]

    if len(whr) == 0:
        # No unused adjacent nodes
        return [path]
    
    # Else grow path by an adjacent node, restricting
    # to type-A vertices if any remain
    whr_A = [ii for ii in whr if typeA[ii]]
    if len(whr_A) > 0:
        whr = whr_A

    paths = [grow_path([*path, ii], adjacency_matrix, typeA) for ii in whr]
    paths = [aa for bb in paths for aa in bb]

    return paths

def knapsack(item_weights, item_values, max_weights):
    """Solves the knapsack problem with given item weights/values and maximum weights.

    For example,
        item_weights = [np.array([1, 0, 3, 0]),
                        np.array([1, 0, 0, 1]),
                        np.array([0, 2, 1, 0])]
        item_values = [20, 10, 5]
        max_weights = np.array([10, 10, 10, 10])

    has optimal solution 140 (corresponding to taking [2, 8, 4] of each item)

    Args:
        item_weights (list): List of ndarrays containing items' vectors of weights.
        item_values (list): List of item values.
        max_weights (ndarray): Array of maximum weights.

    Returns:
        int: Maximum total value.
    """

    # Solve using dynamic programming, starting from empty configuration
    # with zero value. 'values' keeps track of the highest attainable
    # value for a given total item weight.
    empty_config = len(item_weights[0]) * (0,)
    values = {empty_config: 0}

    # Add items to configurations one by one
    for item_weight, item_value in zip(item_weights, item_values):
        # List of newly-attained total weight
        to_add = []

        # Add item to each current optimal configurations
        for config in values:
            n = 1
            done = False
            while not done:

                # Add in n of this item until maximum weight exceeded
                config_new = tuple(np.array(config) + n*item_weight)
                value_new = values[config] + n*item_value
                if any(config_new > max_weights):
                    done = True
                    break

                if config_new in values:
                    # This total weight has already been seen: update optimal value
                    values[config_new] = max(values[config_new], value_new)
                else:
                    # This total weight has not been seen: add to list to add at end
                    # (this avoids revisiting this again later in the 'for config in values' loop)
                    to_add.append([config_new, value_new])

                n += 1

        # Add in newly-achieved total weights
        for config, value in to_add:
            values[config] = value

    # Find the optimal configuration
    config_best = empty_config
    for config in values:
        if values[config] > values[config_best]:
            config_best = config
    
    return values[config_best]
