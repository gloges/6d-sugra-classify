import numpy as np
from itertools import combinations_with_replacement
from tqdm import tqdm
import os

from simpleGroup import SimpleGroup


class Graph():

    def __init__(self, Δ_schedule, T_min_max=np.inf, progress=True, save_folder=None):
        """Initialize a graph for the provided simple groups.

        The simple groups and their corresponding values of `Δ_max` are collected in `Δ_schedule`,
        which should have a nested structure like the following example,
            Δ_schedule = [
                ['A', [[5, 300], [6, 350]]],
                ['B', [[5, 400]]]
            ]
        which specifies A5~SU(6) with `Δ_max=300`, A6~SU(7) with `Δ_max=350` and B5~SO(11) with `Δ_max=400`.

        Args:
            Δ_schedule (list): Nested array of data prescribing the groups' types, ranks and Δ_max.
            T_min_max (int, optional): Upper bound on Tmin. Defaults to np.inf.
            progress (bool, optional): Whether to use progress bars. Defaults to True.
            save_folder (str, optional): Path to folder for saving irreps and vertices. Defaults to None.

        """

        # Save hyper-parameter bound on T_min
        self.T_min_max = T_min_max

        cartan_types = []
        group_ranks  = []
        self.Δs_max  = []
        self.groups  = []

        # Unpack group/Δ data
        for cartan_type, nΔ_pairs in Δ_schedule:            
            cartan_types.extend(len(nΔ_pairs) * [cartan_type])
            group_ranks.extend([n for n, Δmax in nΔ_pairs])
            self.Δs_max.extend([Δmax for n, Δmax in nΔ_pairs])
        
        # Sort by rank *then* alphabetically (i.e. C2, G2, A3, B3, C3, D3,...)
        order = np.lexsort([cartan_types, group_ranks])
        cartan_types = np.array([cartan_types[ii] for ii in order])
        group_ranks  = np.array([group_ranks[ii]  for ii in order])
        self.Δs_max  = [self.Δs_max[ii]  for ii in order]

        # Check for duplicates
        uniq, counts = np.unique([cartan_types, group_ranks], axis=1, return_counts=True)
        for cartan_type, n, count in zip(uniq[0], uniq[1], counts):
            if count > 1:
                err = f'Duplicate group: {cartan_type}[{n}]'
                raise Exception(err)

        # Build the simple groups
        self.groups = [SimpleGroup(cartan_type, n) for cartan_type, n in zip(cartan_types, group_ranks)]

        # Display grid of simple groups
        rank_min = min(group_ranks)
        print(f'\nInitializing graph with {len(self.groups)} simple groups:', flush=True)
        for cartan_type in ['A', 'B', 'C', 'D', 'EFG']:
            whr = np.where([aa in cartan_type for aa in cartan_types])[0]
            if len(whr) > 0:
                ranks = group_ranks[whr]
                print(f'  {cartan_type:>3}:  ', end='', flush=True)
                for n in range(rank_min, max(ranks) + 1):
                    if n in ranks:
                        print(n, end=' ', flush=True)
                    else:
                        print(end=int(np.log10(n)+2)*' ', flush=True)
                print(flush=True)
        print(flush=True)

        # Generate graph data
        # min_irrep_H is the minimal irrep dimension over all irreps (updated while generating vertices)
        self.min_irrep_H = np.inf
        self.generate_vertices(progress, save_folder)
        self.generate_edges(progress)
        
        # Kill off any type-A vertices with too-large Δ
        print('Removing type A vertices with Δ > 273 and no nontrivial edges...', end='', flush=True)
        for vertices in self.vertices:
            vertices = [v for v in vertices
                        if v.type == 'B'
                            or v.Δ <= 273
                            or len(v.edges_nontrivial_A) > 0
                            or len(v.edges_nontrivial_B) > 0]
        print('done.\n', flush=True)

        # Display table of numbers of vertices
        self.display_vertex_counts()

        # Flatten list of vertices, sort (all type-B vertices will be first), and save indices
        self.vertices = np.sort([aa for bb in self.vertices for aa in bb])
        for ii, vertex in enumerate(self.vertices):
            vertex.index = ii

        # Dictionary for storing information about cliques with
        # (Δ+28n) - 0.5*maximum_edge_discount < 0
        self.vertex_neg_values = {}

    def display_vertex_counts(self):
        """For displaying vertex counts during graph initialization."""

        print('[Group | type A | type B]', flush=True)
        print(25*'—', flush=True)

        group_cartan_types = np.unique([group.ID[:1] for group in self.groups if group.ID[:1] < 'E'])
        group_ranks = np.array([group.n for group in self.groups])
        counts = np.zeros([len(group_cartan_types)+1, max(group_ranks)-min(group_ranks)+1, 2], dtype='int')

        # For each group count number of type-A and type-B vertices
        for group, group_vertices in zip(self.groups, self.vertices):
            if group.ID[:1] in ['A', 'B', 'C', 'D']:
                ii = np.where(group.ID[:1] == group_cartan_types)[0][0]
            else:
                ii = len(group_cartan_types)

            jj = group.n - min(group_ranks)

            num_B = 0
            while num_B < len(group_vertices) and group_vertices[num_B].type == 'B':
                num_B += 1
            num_total = len(group_vertices)
            num_A = num_total - num_B

            counts[ii,jj,0] = num_A
            counts[ii,jj,1] = num_B

        # Find max number of digits for aligning counts in table
        max_digs = np.ones([len(counts), 2], dtype='int')
        for ii in range(len(counts)):
            nums_A = max(counts[ii,:,0])
            if nums_A > 0:
                max_digs[ii,0] = int(np.log10(nums_A)) + 1
                max_digs[ii,0] += max_digs[ii,0]//3
                
            nums_B = max(counts[ii,:,1])
            if nums_B > 0:
                max_digs[ii,1] = int(np.log10(nums_B)) + 1
                max_digs[ii,1] += max_digs[ii,1]//3

        # Display counts, with each row corresponding to groups of the same rank
        for ii in range(len(counts[0])):
            n = ii + min(group_ranks)

            for jj, digs in enumerate(max_digs):
                if jj < len(counts) - 1:
                    cartan_type = group_cartan_types[jj]
                elif n == 2:
                    cartan_type = 'G'
                elif n == 4:
                    cartan_type = 'F'
                else:
                    cartan_type = 'E'

                if max(counts[jj,ii]) == 0:
                    print(end=(digs[0]+digs[1]+14)*' ', flush=True)

                else:
                    print('  ' + cartan_type + str(n).rjust(2, '0'), end=':', flush=True)
                    if counts[jj,ii,0] > 0:
                        print(f'{counts[jj,ii,0]:{digs[0]+2},}', end='', flush=True)
                    else:
                        print(end=(digs[0]+2)*' ', flush=True)
                    if counts[jj,ii,1] > 0:
                        print(f'{counts[jj,ii,1]:{digs[1]+2},}', end='', flush=True)
                    else:
                        print(end=(digs[1]+2)*' ', flush=True)
                    print(end='    ', flush=True)

            print(flush=True)
        print(flush=True)

    def generate_vertices(self, progress=True, save_folder=None):
        """Find all solutions of the B-constraint of bounded Δ for each simple group
        and create corresponding vertex objects.

        """

        print('Building vertices...', flush=True)

        self.vertices = []

        for group, Δ_max in zip(self.groups, self.Δs_max):

            # Determine maximum H for this group, given V and Δ bounds, and generate all vertices up to this bound
            H_max_grp = Δ_max + group.V
            group.generate_irreps(H_max_grp)
            self.min_irrep_H = min(self.min_irrep_H, min(group.irreps['H']))
            group_vertices = group.get_vertices(H_max_grp, self.T_min_max, progress)

            # Sort and save!
            # Note: Sorting places in order of increasing bi.bi, which is leveraged when creating edges
            self.vertices.append(np.sort(group_vertices))

            # Save vertices to folder if provided
            if save_folder is not None:
                save_folder_irreps = save_folder + f'/irreps/{group.cartan_type}'
                os.makedirs(save_folder_irreps, exist_ok=True)
                group.save_irreps(save_folder_irreps)

                save_folder_vertices = save_folder + f'/vertices/{group.cartan_type}'
                os.makedirs(save_folder_vertices, exist_ok=True)
                save_vertices_filename = save_folder_vertices + f'/{group.ID}-vertices.tsv'
                self.save_vertices(group_vertices, save_vertices_filename)

        for ii_group, group_vertices in enumerate(self.vertices):
            group = self.groups[ii_group]

            # Identify and remove...
            to_remove = []
            for ii, v in enumerate(group_vertices):

                if (v.Δ < -29):
                    # Vertices with Δ < -29
                    to_remove.append(ii)

                elif v.T_min > self.T_min_max:
                    # Vertices with T_min too large
                    to_remove.append(ii)

                elif (group.cartan_type == 'A' and (group.n+1) % 2 == 0):
                    # Effectively identical vertices
                    # (check that the corresponding Sp group is present)
                    C_present = False
                    for gr in self.groups:
                        if gr.ID[:1] == 'C' and gr.n == (group.n+1)//2:
                            C_present = True
                    if not C_present:
                        continue

                    if v.bibi == -1 and v.b0bi == 1 and v.Δ == (group.n+1 + 7)*(group.n+1 + 8)//2 - 27:
                        to_remove.append(ii)
                    elif v.bibi == 0 and v.b0bi == 2 and v.Δ == 15*(group.n+1) + 1:
                        to_remove.append(ii)

                else:
                    lower_bound_naive = v.Δ + 28*int(v.bibi < 0)
                    for irr in v.irreps:
                        if irr['n'] >= self.min_irrep_H:
                            lower_bound_naive -= irr['n']*irr['H']
                    if lower_bound_naive > 273:
                        to_remove.append(ii)

            if len(to_remove) > 0:
                self.vertices[ii_group] = np.delete(group_vertices, to_remove)

    def generate_edges(self, progress=True):
        """Create all nontrivial edges between vertices in the graph.

        Edges are constructed using different strategies based on the vertex types.
            • A ←→ A: Using Amax and Amax_hat to bound bi.bj and thus bj.bj of candidate vertices.
            • A/B ←→ B: Store type-B vertices based on (n,H) pairs, then loop through all vertices
                        and only check against type-B vertices with viable (n,H).

        Args:
            detailed_progress (bool, optional): Whether to show additional progress bars when constructing vertices and edges (may not work in notebooks). Defaults to True.

        """

        # (type-A ←→ type-A) edges first
        print('Building A ←→ A edges...', flush=True)

        # Iterate over pairs of simple groups (with replacement)
        group_vertex_iterator = combinations_with_replacement(zip(self.groups, self.vertices), r=2)
        
        for [group_1, vertices_1], [group_2, vertices_2] in group_vertex_iterator:
            
            # Iterate over vertices of group_1
            vertex_1_iterator = enumerate(vertices_1)
            if progress:
                vertex_1_iterator = tqdm(vertex_1_iterator, total=len(vertices_1),
                                            desc=f'  {group_1.ID} ←→ {group_2.ID} ({group_1.ID} vertices)',
                                            leave=False, ascii=True)
            
            for ii_1, vertex_1 in vertex_1_iterator:

                # Skip type-B vertices
                if vertex_1.type == 'B':
                    continue

                # Prepare Amax function for this vertex
                group_1.reset_AmaxG(vertex_1, self.min_irrep_H)

                # Iterate over vertices of group_2
                for ii_2, vertex_2 in enumerate(vertices_2):

                    # Skip type-B vertices
                    if vertex_2.type == 'B':
                        continue
                    
                    # Avoid checking pairs more than once: if the vertices have
                    # the same simple group, only need to check v2 ≤ v1
                    if group_1 is group_2 and ii_2 > ii_1:
                        break

                    off_diag_sum = 0
                    for irr in vertex_2.irreps:
                        off_diag_sum += irr['A'] * group_1.get_AmaxG(irr['n'])
                    
                    # Only need to check v2s up to the bound on b2.b2 (vertices are sorted by bibi)
                    if vertex_1.bibi * vertex_2.bibi > off_diag_sum**2:
                        continue

                    # All good! edges may exist
                    vertex_1.construct_edges(vertex_2)

        # (type-A/B ←→ type-B) edges next
        print('Building A/B ←→ B edges...', flush=True)

        # Store all type-B vertices in an array according to their (nR, HR) pairs
        nH_pairs_B = [[]]
        for group_vertices in self.vertices:
            for vertex in group_vertices:
                # All type-B vertices appear before type-A vertices
                if vertex.type == 'A':
                    break

                for irr in vertex.irreps:
                    nR = irr['n']
                    HR = irr['H']

                    # Make sure array is large enough
                    while len(nH_pairs_B)     <= nR: nH_pairs_B.append([])
                    while len(nH_pairs_B[nR]) <= HR: nH_pairs_B[nR].append([])

                    # Store in array
                    nH_pairs_B[nR][HR].append(vertex)

        # Iterate over vertices associated with each simple group
        for group_vertices, group in zip(self.vertices, self.groups):

            # Iterate over vertices for this simple group
            vertex_iterator = group_vertices
            if progress:
                vertex_iterator = tqdm(vertex_iterator, desc=f'  {group.ID} ←→ all ({group.ID} vertices)', leave=False, ascii=True)

            for vertex_1 in vertex_iterator:
                
                # Fetch potential type-B vertices from the (n,H)-array
                vertex_candidates = []
                for irr in vertex_1.irreps:
                    nR = irr['n']
                    HR = irr['H']

                    # nB x (R_B) of a type-B vertex needs to have nB >= H and HB <= n
                    for nR_B in range(HR, len(nH_pairs_B)):
                        for H_B in range(self.min_irrep_H, min(nR+1, len(nH_pairs_B[nR_B]))):
                            vertex_candidates.extend(nH_pairs_B[nR_B][H_B])

                # Remove any duplicates (a type-B vertex can be in the (n,H)-array multiple times)
                vertex_candidates = np.unique(vertex_candidates)

                # If v1 is also of type B, avoid checking the type-B/type-B pair more than once
                if vertex_1.type == 'B':
                    vertex_candidates = [cand for cand in vertex_candidates if not (cand < vertex_1)]

                # Construct edges to all type-B vertex candidates
                for vertex_2 in vertex_candidates:
                    vertex_1.construct_edges(vertex_2)

    def save_vertices(self, vertices, filename):
        """Saves a list of vertices to a file."""

        if len(vertices) == 0:
            return
            
        # Sort by type (B first), Δ then bi.bi
        sort_data = np.array([[vertex.bibi, vertex.Δ, vertex.type == 'A'] for vertex in vertices])
        order = np.lexsort(sort_data.T)
        vertices_sorted = [vertices[ii] for ii in order]

        # Check if file already exists
        Δ_max_saved = -np.inf
        if os.path.isfile(filename):
            with open(filename, 'r') as file:
                for line in file.readlines():
                    Δ_max_saved = max(Δ_max_saved, int(line.split('\t')[1][4:]))

        if vertices_sorted[-1].Δ > Δ_max_saved:
            # Create/overwrite file
            with open(filename, 'w', encoding='utf-8') as file:
                for vertex in vertices_sorted:
                    file.write(vertex.get_save_string() + '\n')

    def update_neg_values(self, clique):
        """Records data about cliques for which f_(1/2) is negative."""

        if clique.num_AB[0] != 0:
            # Only type-B cliques
            return False

        value = clique.f(0.5, False)

        if value >= 0:
            return False
        
        for ii in clique.active_vertices:
            vertex = clique.vertices[ii]
            vertex_irreps_avbl = clique.irreps_avbl[ii]

            # New vertex: add to dictionary
            if vertex.index not in self.vertex_neg_values:
                self.vertex_neg_values[vertex.index] = [[vertex_irreps_avbl, value]]
                continue

            # Vertex appears already: determine if this cliques f(0.5) is lower given irrep availability
            to_add = False
            for v_avbl_current, value_current in self.vertex_neg_values[vertex.index]:
                if value <= value_current or any(vertex_irreps_avbl > v_avbl_current):
                    to_add = True

            if to_add:
                # It's better: scan through data deleting those which are now worse/redundant
                for jj in range(len(self.vertex_neg_values[vertex.index])-1, -1, -1):
                    v_avbl_current, value_current = self.vertex_neg_values[vertex.index][jj]
                    if value <= value_current and all(vertex_irreps_avbl >= v_avbl_current):
                        self.vertex_neg_values[vertex.index].pop(jj)

                # finally, add in new data
                self.vertex_neg_values[vertex.index].append([vertex_irreps_avbl, value])

        return True

    def get_neg_value(self, index, irreps_avbl_min):
        """Fetches information about negative f_(1/2) values for cliques given irrep availability."""

        if index not in self.vertex_neg_values:
            return 0

        value = 0

        for n_avbl, neg_value in self.vertex_neg_values[index]:
            if all(n_avbl >= irreps_avbl_min):
                value = min(value, neg_value)

        return value
