import argparse
import math
from io import StringIO
import numpy as np
from Bio import SeqIO, Phylo
from matplotlib import pyplot as plt

NUCLEOTIDES = ['A', 'C', 'G', 'T']
NT_TO_IDX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}

# ------------------ I/O ------------------ #

def parse_fasta_file(file_path: str):
    """
    Parse a FASTA file into a dictionary {id: sequence}.
    All sequences are assumed to be aligned.
    """
    sequences = {}
    with open(file_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences

def plot_tree(newick_string: str, file_name: str):
    """
    Visualizes a phylogenetic tree given in Newick format and saves it as an image.
    """
    tree = Phylo.read(StringIO(newick_string), "newick")

    def branch_length_formatter(clade):
        if clade.branch_length is not None:
            return f"{clade.branch_length:.3f}"
        return ""

    Phylo.draw(tree, do_show=False, branch_labels=branch_length_formatter)
    plt.title("Phylogenetic Tree")
    plt.savefig(file_name)
    plt.close()


# ------------------ Part 1: Evolutionary Time ------------------ #

def calculate_time_between_two_sequences(seq1, seq2):
    """
    Calculate evolutionary distance between two aligned sequences
    using the Jukes–Cantor (JC69) model.

    Returns:
        float: evolutionary distance
    """
    length = min(len(seq1), len(seq2))
    if length == 0:
        return 0.0

    mismatches = sum(1 for i in range(length) if seq1[i] != seq2[i])
    D = mismatches / length

    # Handle the boundary case where D is too large for Jukes-Cantor
    if D >= 0.75:
        D = 0.749999

    t = -0.75 * math.log(1.0 - (4.0 / 3.0) * D)
    return t


# ------------------ Part 2: Distance Matrix ------------------ #

def calculate_distance_matrix(sequences):
    """
    Compute a pairwise distance matrix for all sequences
    using the Jukes–Cantor model.

    Returns:
        ids (list)
        distance_matrix (np.ndarray)
    """
    ids = list(sequences.keys())
    # Using dict of dicts to preserve the structure and as stated in the exercise
    distance_matrix = {a: {} for a in ids}

    for i in range(len(ids)):
        for j in range(i, len(ids)):
            seq_i = sequences[ids[i]]
            seq_j = sequences[ids[j]]
            dist_ij = calculate_time_between_two_sequences(seq_i, seq_j)
            distance_matrix[ids[i]][ids[j]] = dist_ij
            distance_matrix[ids[j]][ids[i]] = dist_ij

    return ids, distance_matrix


# ------------------ Part 3: Tree Construction ------------------ #

def neighbor_joining(distance_matrix, ids):
    """
    Build a phylogenetic tree using the Neighbor Joining algorithm.

    Steps:
    1. Compute total divergence r_i
    2. Compute adjusted matrix M_ij
    3. Join the closest pair
    4. Update distances
    5. Repeat until two nodes remain

    Returns:
        str: Newick formatted tree
    """
    trees = {seq: seq for seq in ids}
    working_set = ids.copy()

    while len(working_set) > 2:
        n = len(working_set)

        # 1. Calculate net divergence (u) for each active node
        net_div = {}
        for node in working_set:
            total_dist = sum(distance_matrix[node][other] for other in working_set if other != node)
            net_div[node] = total_dist / (n - 2)

        # 2. Find the pair with the lowest adjusted distance score
        best_pair = None
        lowest_score = float('inf')

        for i in range(n):
            for j in range(i + 1, n):
                n1, n2 = working_set[i], working_set[j]

                # Dynamic Q-value equivalent calculation
                score = distance_matrix[n1][n2] - net_div[n1] - net_div[n2]

                if score < lowest_score:
                    lowest_score = score
                    best_pair = (n1, n2)

        x, y = best_pair

        # 3. Calculate new branch lengths
        dist_xy = distance_matrix[x][y]
        branch_x = 0.5 * dist_xy + 0.5 * (net_div[x] - net_div[y])
        branch_y = 0.5 * dist_xy + 0.5 * (net_div[y] - net_div[x])

        # 4. Create the new internal node
        joined_node = f"({trees[x]}:{branch_x:.5f},{trees[y]}:{branch_y:.5f})"

        # 5. Update the distance matrix with the new node
        distance_matrix[joined_node] = {}
        for item in working_set:
            if item not in (x, y):
                new_d = 0.5 * (distance_matrix[x][item] + distance_matrix[y][item] - dist_xy)
                distance_matrix[joined_node][item] = new_d
                distance_matrix[item][joined_node] = new_d

        # 6. Housekeeping: Update active sets
        trees[joined_node] = joined_node
        working_set.remove(x)
        working_set.remove(y)
        working_set.append(joined_node)

    # Final join for the remaining two nodes
    left, right = working_set
    final_dist = distance_matrix[left][right]
    return f"({trees[left]}:{final_dist / 2:.5f},{trees[right]}:{final_dist / 2:.5f});"


def UPGMA(distance_matrix, ids):
    """
    Build a phylogenetic tree using the UPGMA algorithm.

    Notes:
    - Assumes a molecular clock
    - Distances must be ultrametric

    Returns:
        str: Newick formatted tree
    """
    # Dictionary tracking: cluster_name -> (Newick_string, leaf_count)
    cluster_data = {leaf: (leaf, 1) for leaf in ids}
    active_nodes = ids.copy()

    while len(active_nodes) > 1:
        # 1. Locate the two closest clusters
        shortest_dist = float('inf')
        merge_candidates = (None, None)

        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                c1, c2 = active_nodes[i], active_nodes[j]
                d = distance_matrix[c1][c2]
                if d < shortest_dist:
                    shortest_dist = d
                    merge_candidates = (c1, c2)

        node_a, node_b = merge_candidates
        tree_a, size_a = cluster_data[node_a]
        tree_b, size_b = cluster_data[node_b]

        # 2. Form the new merged cluster
        branch_len = shortest_dist / 2.0
        merged_name = f"({tree_a}:{branch_len:.5f},{tree_b}:{branch_len:.5f})"
        total_size = size_a + size_b
        cluster_data[merged_name] = (merged_name, total_size)

        # 3. Calculate weighted average linkages to all other remaining nodes
        distance_matrix[merged_name] = {}
        for target in active_nodes:
            if target not in (node_a, node_b):
                weight_a = distance_matrix[node_a][target] * size_a
                weight_b = distance_matrix[node_b][target] * size_b
                avg_dist = (weight_a + weight_b) / total_size

                distance_matrix[merged_name][target] = avg_dist
                distance_matrix[target][merged_name] = avg_dist

        # 4. Clean up active tracking
        active_nodes.remove(node_a)
        active_nodes.remove(node_b)
        active_nodes.append(merged_name)

    return cluster_data[active_nodes[0]][0] + ";"


# ------------------ Part 4: Felsenstein Pruning ------------------ #

def jukes_cantor_transition_prob(t):
    """
    Compute JC69 transition probabilities for branch length t.

    Returns:
        dict mapping (parent_nt, child_nt) -> probability
    """
    # Vectorized matrix construction
    P = np.zeros((4, 4))
    p_no_change = 0.25 + 0.75 * math.exp(-(4.0 / 3.0) * t)
    p_change = 0.25 - 0.25 * math.exp(-(4.0 / 3.0) * t)

    np.fill_diagonal(P, p_no_change)
    P[P == 0] = p_change
    return P


def felsenstein_pruning(newick_string: str, sequences: dict):
    """
    Compute the log-likelihood of a tree using Felsenstein's pruning algorithm
    and reconstruct the most likely root sequence.

    Returns:
        tuple:
            reconstructed_root_sequence (str)
            log_likelihood (float)
    """
    parsed_tree = Phylo.read(StringIO(newick_string), "newick")
    seq_length = len(list(sequences.values())[0])

    # Dynamic Programming table: Maps a node to its likelihood array (seq_length x 4)
    dp_likelihoods = {}

    # Traverse tree in post-order (bottom-up)
    for node in parsed_tree.find_clades(order="postorder"):
        dp_likelihoods[node] = np.zeros((seq_length, 4))

        if node.is_terminal():
            # Init leaves
            leaf_seq = sequences[node.name]
            for pos, nucleotide in enumerate(leaf_seq):
                if nucleotide in NT_TO_IDX:
                    dp_likelihoods[node][pos, NT_TO_IDX[nucleotide]] = 1.0
        else:
            # Init internal nodes
            for pos in range(seq_length):
                for base_idx in range(4):
                    prob_product = 1.0
                    for child_node in node.clades:
                        branch_len = child_node.branch_length or 0.0
                        trans_matrix = jukes_cantor_transition_prob(branch_len)

                        # Use numpy dot product logic for cleaner mathematical calculation
                        prob_product *= np.sum(trans_matrix[base_idx, :] * dp_likelihoods[child_node][pos, :])

                    dp_likelihoods[node][pos, base_idx] = prob_product

    # Root reconstruction phase
    final_log_lik = 0.0
    ancestor_seq = ""
    prior = 0.25  # Assumes equal base frequencies

    root_node = parsed_tree.root
    for pos in range(seq_length):
        marginals = dp_likelihoods[root_node][pos, :] * prior
        site_prob = np.sum(marginals)

        if site_prob > 0:
            final_log_lik += math.log(site_prob)

        best_base_idx = np.argmax(marginals)
        ancestor_seq += NUCLEOTIDES[best_base_idx]

    return ancestor_seq, final_log_lik


# ------------------ Command Line Interface ------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phylogenetics Exercise")

    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to aligned FASTA file"
    )

    parser.add_argument(
        "--part",
        type=str,
        required=True,
        choices=["build_tree", "likelihood", "reconstruction"],
        help="Which analysis step to run"
    )

    parser.add_argument(
        "--tree_type",
        type=str,
        choices=["NJ", "UPGMA"],
        help="Tree construction method (required for build_tree, likelihood, reconstruction)"
    )

    args = parser.parse_args()
    sequences = parse_fasta_file(args.fasta)

    if args.part == "build_tree":
        if args.tree_type is None:
            raise ValueError("--tree_type is required for build_tree")

        ids, dist_matrix = calculate_distance_matrix(sequences)

        if args.tree_type == "NJ":
            tree = neighbor_joining(dist_matrix, ids)
        else:
            tree = UPGMA(dist_matrix, ids)

        print(tree)
        plot_tree(tree, "cool_tree.png")

    elif args.part in ["likelihood", "reconstruction"]:
        if args.tree_type is None:
            raise ValueError("--tree_type is required")

        ids, dist_matrix = calculate_distance_matrix(sequences)

        if args.tree_type == "NJ":
            tree = neighbor_joining(dist_matrix, ids)
        else:
            tree = UPGMA(dist_matrix, ids)

        print(tree)
        plot_tree(tree, "cool_tree.png")

        root_seq, log_likelihood = felsenstein_pruning(tree, sequences)

        if args.part == "likelihood":
            print("Log-Likelihood:", log_likelihood)
        else:
            print("Reconstructed Root Sequence:", root_seq)
