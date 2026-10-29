import os
import gzip
import sys
import numpy as np
from Bio import SeqIO


# =====================================================
# import matplotlib.pyplot as plt  ####### TO RUN THIS FILE INSTALL matplotlib AND COMMENT OUT THIS LINE
# =====================================================


# Dev-only constants
MODEL_TYPE = "markov_llr"  # Available Model Types Implemented: ["hmm_2state", "markov_llr"]
LLR_WINDOW = 175
DEBUG_TRAIN_LIMIT = 50  # how many train sequences to use for fast checks
DEBUG_DEV_LIMIT = 50  # how many dev sequences to use for fast checks
RUN_COMPARISON_PLOTS = False  # set True to generate plots, if True exits process at end
DEBUG_SMALL = True  # set to True to run faster smaller training runs, if True exits process at end
PLOTS_DIR = "plots"


# =====================================================
#                   DATA PROCESSING
# =====================================================
def parse_fasta_file(file_path: str):
    """
    Parses a FASTA file (plain or gzipped) and returns a mapping of sequence identifiers to nucleotide sequences.

    Parameters:
        file_path (str): The path to the FASTA file.

    Returns:
        dict: A dictionary with sequence IDs as keys and DNA sequences as values.
    """
    sequences = {}

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)
    else:
        with open(file_path, 'r') as file_handle:
            for record in SeqIO.parse(file_handle, "fasta"):
                sequences[record.id] = str(record.seq)

    return sequences


def prepare_training_data(sequence_file: str, label_file: str):
    """
    Aligns nucleotide sequences with corresponding labels to create a training dataset.

    Parameters:
        sequence_file (str): Path to the FASTA file containing sequences.
        label_file (str): Path to the FASTA file containing labels.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains a DNA sequence and its label.
    """
    sequences = parse_fasta_file(sequence_file)
    labels = parse_fasta_file(label_file)

    if sequences.keys() != labels.keys():
        raise ValueError("Mismatch between sequence IDs and label IDs in the provided files.")

    return [(sequences[seq_id], labels[seq_id]) for seq_id in sequences]


def _split_data(pairs, train_frac=0.6, dev_frac=0.2, seed=42):
    """
    Split a list of (sequence, label) pairs into train/dev/test.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    n = len(pairs)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)

    train_idx = indices[:n_train]  # [0,...,train_frac*n]
    dev_idx = indices[n_train:n_train + n_dev]  # [train_frac*n,...,(train_frac+dev_frac)*n]
    test_idx = indices[n_train + n_dev:]  # [(train_frac+dev_frac)*n,...,n]
    train = [pairs[i] for i in train_idx]
    dev = [pairs[i] for i in dev_idx]
    test = [pairs[i] for i in test_idx]

    assert len(train) + len(dev) + len(test) == len(pairs)  # sanity check to see if we used all data...
    return train, dev, test


def _reverse_complement(seq: str) -> str:
    """
    Compute the reverse complement of a DNA sequence (A,C,G,T,N).
    """
    comp = str.maketrans({
        'A': 'T', 'T': 'A',
        'C': 'G', 'G': 'C',
        'a': 't', 't': 'a',
        'c': 'g', 'g': 'c',
        'N': 'N', 'n': 'n'
    })
    return seq.translate(comp)[::-1]


def _augment_with_reverse_complement(pairs):
    """
    Given a list of (sequence, label) pairs, return an augmented list
    including reverse-complemented sequences with reversed labels.
    """
    augmented = list(pairs)
    for seq, lbl in pairs:
        rc_seq = _reverse_complement(seq)
        rc_lbl = lbl[::-1]  # CpG island labels just reverse
        augmented.append((rc_seq, rc_lbl))
    return augmented


# =====================================================
#                   MODEL TRAINING
# =====================================================
def _train_hmm_2state(training_data):
    """
    Trains a simple 2-state HMM:
      States: 'N' (non-island), 'C' (CpG island)
      Nucleotides: 'A', 'C', 'G', 'T'

    Parameters:
        training_data (list[tuple[str, str]]): (sequence, label) pairs.

    Returns:
        dict: HMM parameters stored in a dictionary with log-probabilities.
    """

    states = ['N', 'C']  # hidden states
    symbols = ['A', 'C', 'G', 'T']  # nucleotides, symbols for the observations

    state_to_idx = {s: i for i, s in enumerate(states)}  # convert each state to an index
    sym_to_idx = {x: i for i, x in enumerate(symbols)}  # convert each symbol to an index

    num_states = len(states)  # total num of states in the model
    num_syms = len(symbols)  # total num of differnt symbols in the model

    # Laplace smoothing: start with 1 instead of 0
    init_counts = np.ones(num_states)  # P(S0)
    trans_counts = np.ones((num_states, num_states))  # P(S_t | S_{t-1})
    emit_counts = np.ones((num_states, num_syms))  # P(X_t | S_t)

    for seq, lbl in training_data:
        if not seq or not lbl:
            continue
        if len(seq) != len(lbl):
            continue

        # First position: initial state + emission
        s0 = lbl[0]
        if s0 in state_to_idx:
            s0_idx = state_to_idx[s0]
            init_counts[s0_idx] += 1

            x0 = seq[0].upper()
            if x0 in sym_to_idx:
                emit_counts[s0_idx, sym_to_idx[x0]] += 1

        # Remaining positions: transitions + emissions
        for prev_state_char, curr_state_char, base in zip(lbl[:-1], lbl[1:], seq[1:]):  # Hi-1, Hi, Xi
            if prev_state_char not in state_to_idx or curr_state_char not in state_to_idx:
                continue
            # Update states tranmission counts
            i = state_to_idx[prev_state_char]  # i = Hi-1
            j = state_to_idx[curr_state_char]  # j = Hi
            trans_counts[i, j] += 1  # t[i,j] = t[Hi-1,Hi]

            # Update observation|state emission counts
            base_u = base.upper()  # Xi ('A' or 'C' or 'G' or 'T')
            if base_u in sym_to_idx:
                emit_counts[j, sym_to_idx[base_u]] += 1  # π[j,base] = π[Xi,Hi]

    # Normalize to get probabilities
    init_probs = init_counts / init_counts.sum()
    trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    emit_probs = emit_counts / emit_counts.sum(axis=1, keepdims=True)

    # Store in log-space for numerical stability
    model = {
        "model_type": "hmm_2state",
        "states": states,
        "symbols": symbols,
        "state_to_idx": state_to_idx,
        "sym_to_idx": sym_to_idx,
        "log_init": np.log(init_probs),
        "log_trans": np.log(trans_probs),
        "log_emit": np.log(emit_probs),
    }
    return model


def _train_markov_llr(training_data, window: int = 200):
    """
    Train two separate 1st-order Markov chains over nucleotides:
      - Chain 0 models background (positions labeled 'N')
      - Chain 1 models CpG islands (positions labeled 'C')

    For each chain we learn:
      - P(x_1 | class)      : initial nucleotide distribution
      - P(x_t | x_{t-1}, class) : transition probabilities between nucleotides
      - A prior P(class)    : how often 'N' vs 'C' appears in the data overall

    The 'window' parameter is *not* used during training itself.
    We just store it in the model so that later, during annotation, we know
    how wide a window to look at around each position when computing LLRs (Log-Likelihood Ratio).
    """

    # Alphabet of allowed nucleotides
    symbols = ['A', 'C', 'G', 'T']
    sym_to_idx = {x: i for i, x in enumerate(symbols)}
    M = len(symbols)

    # We have exactly 2 classes:
    #   0 -> No island 'N'
    #   1 -> CpG island 'C'
    K = 2

    # Laplace-smoothing:
    # Start with all ones so no probability is ever exactly zero.
    # init_counts[k, m]   counts how often nucleotide m starts a run of class k
    # trans_counts[k, i, j] counts how often we see i -> j transitions in class k
    init_counts = np.ones((K, M))  # shape (2, 4)
    trans_counts = np.ones((K, M, M))  # shape (2, 4, 4)

    # label_counts[k] counts how many positions are labeled as that class,
    # used to estimate P(class = k)
    label_counts = np.zeros(K)

    # Go over all (sequence, label) pairs
    for seq, lbl in training_data:
        # Skip bad pairs
        if not seq or not lbl or len(seq) != len(lbl):
            continue

        seq_u = seq.upper()
        L = len(seq_u)

        # --- 1) Count how often each label appears overall (for priors) ---
        for ch in lbl:
            if ch == 'N':
                label_counts[0] += 1
            elif ch == 'C':
                label_counts[1] += 1

        # --- 2) Break the sequence into contiguous runs of 'N' or 'C' ---
        # We will treat each run separately and update the Markov statistics
        # according to the class of that run.
        i = 0
        while i < L:
            lab = lbl[i]
            # Ignore labels that are not 'N' or 'C'
            if lab not in ('N', 'C'):
                i += 1
                continue

            # Decide which class index this run belongs to
            class_idx = 0 if lab == 'N' else 1

            # Find end index j of this contiguous run of the same label
            j = i + 1
            while j < L and lbl[j] == lab:
                j += 1

            # This is the subsequence corresponding to that run
            run = seq_u[i:j]

            # --- 3) Update initial + transition counts for this class ---
            prev_idx = None
            first_valid_done = False

            for base in run:
                # Ignore any base not in {A, C, G, T}
                if base not in sym_to_idx:
                    prev_idx = None  # break the Markov chain through here
                    continue

                b_idx = sym_to_idx[base]

                # First *valid* base in this run -> contributes to initial distribution
                if not first_valid_done:
                    init_counts[class_idx, b_idx] += 1
                    first_valid_done = True

                # Every pair of consecutive valid bases -> transition from prev_idx to b_idx
                if prev_idx is not None:
                    trans_counts[class_idx, prev_idx, b_idx] += 1

                prev_idx = b_idx

            # Move to next run
            i = j

    # --- 4) Convert label_counts to priors P(class) ---
    total_labels = label_counts.sum()
    if total_labels > 0:
        priors = label_counts / total_labels
    else:
        # Fallback if no labels encountered (shouldn't really happen here)
        priors = np.array([0.5, 0.5])
    # Add a tiny epsilon before log to avoid log(0)
    log_prior = np.log(priors + 1e-12)

    # --- 5) Normalize counts to get probabilities ---
    # For each class, divide init_counts row by its sum -> P(x_1 | class)
    init_probs = init_counts / init_counts.sum(axis=1, keepdims=True)

    # For each class and previous symbol, normalize transitions over next symbol
    # trans_probs[k, i, j] = P(next = j | prev = i, class = k)
    trans_probs = trans_counts / trans_counts.sum(axis=2, keepdims=True)

    # Take logs for numerical stability
    log_init = np.log(init_probs)
    log_trans = np.log(trans_probs)

    # --- 6) Pack everything into a model dictionary ---
    model = {
        "model_type": "markov_llr",
        "symbols": symbols,
        "sym_to_idx": sym_to_idx,
        # The two Markov chains:
        #   chain 0 = background 'N'
        #   chain 1 = CpG island 'C'
        "log_init": log_init,  # shape (2, 4)
        "log_trans": log_trans,  # shape (2, 4, 4)
        "log_prior": log_prior,  # shape (2,), log P(class)
        "window": window,  # how wide a window to use at prediction time
    }
    return model


def train_classifier(training_data):
    """
    Trains a classifier to identify CpG islands in DNA sequences.
    Entry point - decides which model to train based on constant MODEL_TYPE.

    Parameters:
        training_data (list[tuple[str, str]]): Training data consisting of sequences and their labels.

    Returns:
        object: A trained classifier model according to MODEL_TYPE.
    """

    if MODEL_TYPE == "hmm_2state":
        return _train_hmm_2state(training_data)
    elif MODEL_TYPE == "markov_llr":
        return _train_markov_llr(training_data, LLR_WINDOW)
    # elif MODEL_TYPE == "hmm_4state":
    #     return train_hmm_4state(training_data)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


# =====================================================
#                   MODEL PREDICTION
# =====================================================
def _viterbi_decode(model, sequence: str) -> str:
    """
    Runs Viterbi decoding on a DNA sequence using a 2-state HMM model.

    Returns:
        str: Most likely state path as a string over 'N'/'C'.
    """
    states = model["states"]
    symbols = model["symbols"]
    state_to_idx = model["state_to_idx"]
    sym_to_idx = model["sym_to_idx"]

    log_init = model["log_init"]  # shape: (K,)
    log_trans = model["log_trans"]  # shape: (K, K)
    log_emit = model["log_emit"]  # shape: (K, M)

    K = len(states)
    T = len(sequence)

    if T == 0:
        return ""

    # DP tables
    dp = np.full((K, T), -np.inf)  # dp[k, t] = best log-prob of path ending in state k at position t
    backptr = np.zeros((K, T), dtype=int)

    # Helper for unknown nucleotides (e.g. 'N'): treat as uninformative
    uniform_emit_log = -np.log(len(symbols))  # uniform distribution

    # Initialization (t = 0)
    x0 = sequence[0].upper()
    if x0 in sym_to_idx:
        emit0 = log_emit[:, sym_to_idx[x0]]
    else:
        emit0 = np.full(K, uniform_emit_log)

    dp[:, 0] = log_init + emit0

    # Recursion
    for t in range(1, T):
        base = sequence[t].upper()
        if base in sym_to_idx:
            emit_vec = log_emit[:, sym_to_idx[base]]  # shape: (K,)
        else:
            emit_vec = np.full(K, uniform_emit_log)

        for k in range(K):
            # best previous state for going into k
            prev_scores = dp[:, t - 1] + log_trans[:, k]
            best_prev = np.argmax(prev_scores)
            dp[k, t] = prev_scores[best_prev] + emit_vec[k]
            backptr[k, t] = best_prev

    # Termination: best final state
    last_state = np.argmax(dp[:, T - 1])

    # Backtrace
    state_path_idx = [last_state]
    for t in range(T - 1, 0, -1):
        last_state = backptr[last_state, t]
        state_path_idx.append(last_state)
    state_path_idx.reverse()

    # Convert indices to 'N'/'C'
    state_path = ''.join(states[i] for i in state_path_idx)
    return state_path


def _log_prob_markov_segment(model: dict, sequence: str, chain_idx: int, start: int, end: int) -> float:
    """
    Compute the log-probability of a substring sequence[start:end]
    under one of the Markov chains in the model.

    - chain_idx = 0 -> background chain ('N')
    - chain_idx = 1 -> island chain ('C')

    We:
      * ignore characters that are not A/C/G/T
      * use the learned initial distribution for the first valid base
      * use the learned transition matrix for each subsequent valid base
    """

    symbols = model["symbols"]
    sym_to_idx = model["sym_to_idx"]
    # Initial log-probs for this class: log P(x_1 | class)
    log_init = model["log_init"][chain_idx]  # shape (M,)
    # Transition log-probs for this class: log P(x_t | x_{t-1}, class)
    log_trans = model["log_trans"][chain_idx]  # shape (M, M)

    # Extract and uppercase the substring
    seq_seg = sequence[start:end].upper()

    log_p = 0.0  # will accumulate log-probabilities
    prev_idx = None  # index of previous valid base (None means "no previous")

    for base in seq_seg:
        # Skip non-ACGT bases
        if base not in sym_to_idx:
            prev_idx = None  # break the chain here
            continue

        b_idx = sym_to_idx[base]

        if prev_idx is None:
            # First valid base in this segment:
            # use initial distribution log P(base | class)
            log_p += log_init[b_idx]
        else:
            # Subsequent base: use transition log P(base | prev_base, class)
            log_p += log_trans[prev_idx, b_idx]

        prev_idx = b_idx

    # If there were no valid bases at all, log_p stays 0.0 (effectively neutral)
    return log_p


def _annotate_with_markov_llr(model: dict, sequence: str) -> str:
    """
    Use the trained Markov LLR model to label each position in the sequence as
    'C' (CpG island) or 'N' (background).

    For each position i:
      1. Take a window centered at i (size given by model["window"]).
      2. Compute log P(window | island chain) + log P(island).
      3. Compute log P(window | background chain) + log P(background).
      4. LLR(i) = [island log-prob] - [background log-prob].
      5. If LLR(i) > 0 -> predict 'C' else predict 'N'.

    So a positive LLR means the window around position i looks more like
    CpG-island statistics; a negative LLR means it looks more like background.
    """

    L = len(sequence)
    if L == 0:
        return ""

    # How wide the window is (number of positions we look at around each i)
    # model["window"] should have been set at training time;
    # 200 here is just a fallback default if it wasn't.
    window = model.get("window", 200)
    half = window // 2

    # log_prior[0] = log P(background), log_prior[1] = log P(island)
    log_prior = model["log_prior"]

    out = []  # will collect 'N' / 'C' predictions for each position

    for i in range(L):
        # Define window boundaries around i, clipped to sequence bounds
        start = max(0, i - half)
        end = min(L, i + half + 1)

        # Compute log-prob under each class:
        # chain 0 = background 'N'
        # chain 1 = island 'C'
        log_p_N = _log_prob_markov_segment(model, sequence, 0, start, end)
        log_p_C = _log_prob_markov_segment(model, sequence, 1, start, end)

        # Add priors to get posterior (up to a constant factor in log-space)
        #   log P(window, class) = log P(window | class) + log P(class)
        score_N = log_p_N + log_prior[0]
        score_C = log_p_C + log_prior[1]

        # Log-likelihood ratio:
        #   LLR = log P(window, island) - log P(window, background)
        llr = score_C - score_N

        # If CpG-island score is higher -> predict 'C', else 'N'
        out.append('C' if llr > 0.0 else 'N')

    # Join all predicted characters into one label string
    return ''.join(out)


def annotate_sequence(model: dict, sequence):
    """
    Annotates a DNA sequence with CpG island predictions.

    Parameters:
        model (object): Your trained classifier model.
        sequence (str): A DNA sequence to be annotated.

    Returns:
        str: A string of annotations, where 'C' marks a CpG island region and 'N' denotes non-CpG regions.
    """
    model_type = model.get("model_type")
    if model_type == "hmm_2state":
        return _viterbi_decode(model, sequence)
    elif model_type == "markov_llr":
        return _annotate_with_markov_llr(model, sequence)
    # elif model["model_type"] == "markov_llr":
    #     return annotate_with_markov_llr(model, sequence)
    else:
        # simple fallback or error
        return ''.join('C' if nt.upper() == 'C' else 'N' for nt in sequence)


# =====================================================
#                   MODEL EVALUATION
# =====================================================
def _evaluate_on_pairs(model, data_pairs):
    """
    Evaluate per-base accuracy of the model on a list of (sequence, label) pairs.
    """
    total = 0
    correct = 0

    for idx, (seq, lbl) in enumerate(data_pairs):
        if idx % 50 == 0:
            print(f"  evaluating sequence {idx + 1}/{len(data_pairs)}...")

        # skip sequences that are all 'N'
        if set(seq.upper()) <= {"N"}:
            continue

        pred = annotate_sequence(model, seq)
        L = min(len(pred), len(lbl))
        total += L
        for p, g in zip(pred[:L], lbl[:L]):
            if p == g:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Per-base accuracy: {accuracy:.4f} ({correct}/{total})\n")
    return accuracy


def _majority_baseline(data_pairs, name: str = "Dev"):
    """
    Compute the majority-class baseline on a list of (sequence, label) pairs.
    Returns (majority_label, accuracy).
    A simple baseline model for evaluating other models against.
    """
    counts = {"N": 0, "C": 0}
    total = 0

    for _, lbl in data_pairs:
        for ch in lbl:
            if ch in counts:
                counts[ch] += 1
                total += 1

    if total == 0:
        print(f"{name} majority baseline: no labels found.")
        return None, 0.0

    if counts["N"] >= counts["C"]:
        majority_label = "N"
    else:
        majority_label = "C"

    correct = counts[majority_label]
    acc = correct / total
    # print(
    #     f"{name} majority baseline (always '{majority_label}'): "
    #     f"accuracy = {acc:.4f} ({correct}/{total})"
    # )
    return majority_label, acc


def _finalize_stats(name, correct, total, true_C, pred_C):
    """
    Given raw counts, compute accuracy and 'C' fractions, print a summary,
    and return a dict with the stats.
    """
    accuracy = correct / total if total > 0 else 0.0
    true_C_frac = true_C / total if total > 0 else 0.0
    pred_C_frac = pred_C / total if total > 0 else 0.0

    print(
        f"[{name}] accuracy={accuracy:.4f}, true C frac={true_C_frac:.4f}, "
        f"pred C frac={pred_C_frac:.4f} (correct={correct}/{total})"
    )

    return {
        "name": name,
        "accuracy": accuracy,
        "true_C_frac": true_C_frac,
        "pred_C_frac": pred_C_frac,
        "correct": correct,
        "total": total,
    }


def _compute_model_stats(model, data_pairs, name="model"):
    """
    Compute:
      - per-base accuracy
      - true fraction of bases labeled 'C'
      - predicted fraction of bases labeled 'C'
    on the given (sequence, label) pairs.

    Uses the same skipping rule as _evaluate_on_pairs:
      - skip sequences whose nucleotide sequence is all 'N'.
    """
    total = 0  # total positions considered
    correct = 0  # positions where prediction == gold
    true_C = 0  # positions where gold label == 'C'
    pred_C = 0  # positions where predicted label == 'C'

    for seq, lbl in data_pairs:
        # skip sequences that are all 'N' (like _evaluate_on_pairs)
        if set(seq.upper()) <= {"N"}:
            continue

        pred = annotate_sequence(model, seq)
        L = min(len(pred), len(lbl))

        for p, g in zip(pred[:L], lbl[:L]):
            total += 1
            if p == g:
                correct += 1
            if g == 'C':
                true_C += 1
            if p == 'C':
                pred_C += 1
    return _finalize_stats(name, correct, total, true_C, pred_C)


def _compute_baseline_stats(data_pairs, majority_label='N', name="baseline"):
    """
    Compute accuracy and 'C' fractions for a constant baseline that always predicts
    majority_label ('N' or 'C'), using the same per-base evaluation as the models.

    We skip sequences whose sequence is all 'N', just like in _evaluate_on_pairs.
    """
    total = 0
    correct = 0
    true_C = 0
    pred_C = 0  # will be >0 only if majority_label == 'C'

    for seq, lbl in data_pairs:
        if set(seq.upper()) <= {"N"}:
            continue

        for g in lbl:
            if g not in ('N', 'C'):
                continue
            total += 1
            if g == majority_label:
                correct += 1
            if g == 'C':
                true_C += 1
            if majority_label == 'C':
                pred_C += 1

    return _finalize_stats(name, correct, total, true_C, pred_C)


def run_dev_comparison_plots(train_data, dev_data):
    """
    Train both models (HMM 2-state and Markov LLR) on the same augmented train data
    and generate comparison plots on the dev set:
      1) Dev accuracy per model (baseline, HMM, Markov LLR)
      2) True vs predicted 'C' fraction per model
    """

    # Use the same augmented train data for both models
    train_aug = _augment_with_reverse_complement(train_data)

    print("\n[run_dev_comparison_plots] Training HMM 2-state on augmented train data...")
    hmm_model = _train_hmm_2state(train_aug)

    print("[run_dev_comparison_plots] Training Markov LLR on augmented train data...")
    llr_model = _train_markov_llr(train_aug, window=LLR_WINDOW)

    # Baseline on dev to get majority label; we won't rely on its printed accuracy,
    # because we'll recompute accuracy with the same skipping rule as models.
    print("[run_dev_comparison_plots] Training Majority-Class Baseline on augmented train data...")
    majority_label, _ = _majority_baseline(dev_data, name="Dev")

    print("\n[run_dev_comparison_plots] Computing dev stats for baseline, HMM, and Markov LLR...")

    baseline_stats = _compute_baseline_stats(dev_data, majority_label=majority_label, name="Baseline")
    hmm_stats = _compute_model_stats(hmm_model, dev_data, name="HMM-2state")
    llr_stats = _compute_model_stats(llr_model, dev_data, name="Markov-LLR")

    # ---------- Plot 1: Dev accuracy per model ----------
    model_names = [baseline_stats["name"], hmm_stats["name"], llr_stats["name"]]
    dev_accs = [baseline_stats["accuracy"], hmm_stats["accuracy"], llr_stats["accuracy"]]

    plt.figure()
    plt.bar(model_names, dev_accs)
    plt.ylabel("Dev accuracy")
    plt.title("Dev accuracy per model")
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dev_accuracy_per_model.png"))
    plt.close()

    # ---------- Plot 2: True vs Predicted 'C' fraction ----------
    x = np.arange(len(model_names))
    width = 0.35

    true_C_fracs = [
        baseline_stats["true_C_frac"],
        hmm_stats["true_C_frac"],
        llr_stats["true_C_frac"],
    ]
    pred_C_fracs = [
        baseline_stats["pred_C_frac"],
        hmm_stats["pred_C_frac"],
        llr_stats["pred_C_frac"],
    ]

    plt.figure()
    plt.bar(x - width / 2, true_C_fracs, width, label="True 'C' fraction")
    plt.bar(x + width / 2, pred_C_fracs, width, label="Predicted 'C' fraction")
    plt.xticks(x, model_names)
    plt.ylabel("Fraction of positions labeled 'C'")
    plt.title("True vs predicted CpG island fraction on dev")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "true_vs_pred_c_fraction.png"))
    plt.close()


def debugging_and_dev(train_data, dev_data):
    """
    Run quick experiments on small train/dev subsets for multiple LLR window sizes.
    Prints accuracies and also plots how window size affects train/dev accuracy.
    """
    # Take small subsets
    debug_train = train_data[:DEBUG_TRAIN_LIMIT]
    debug_dev = dev_data[:DEBUG_DEV_LIMIT]
    debug_train = _augment_with_reverse_complement(debug_train)

    # Try several window sizes quickly
    debug_windows = [50, 100, 150, 175, 200]
    dev_accs = []
    train_accs = []

    for w in debug_windows:
        print(f"\n=== Debug run for window={w} ===")
        print(f"Using {len(debug_train)} train sequences and {len(debug_dev)} dev sequences.\n")

        # Train Markov LLR model with this window on the *small* train subset
        debug_model = _train_markov_llr(debug_train, window=w)

        print("Train (subset) accuracy:")
        train_acc = _evaluate_on_pairs(debug_model, debug_train)
        train_accs.append(train_acc)

        print("Dev (subset) accuracy:")
        dev_acc = _evaluate_on_pairs(debug_model, debug_dev)
        dev_accs.append(dev_acc)

    # --------- Plot: window size vs accuracy ---------
    plt.figure()
    plt.plot(debug_windows, train_accs, marker='o', linestyle='--', label='Train (subset)')
    plt.plot(debug_windows, dev_accs, marker='s', linestyle='-', label='Dev (subset)')
    plt.xlabel("LLR window size")
    plt.ylabel("Per-base accuracy")
    plt.title("Effect of LLR window size on Markov model performance (subset)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "window_vs_accuracy_debug.png"))
    plt.close()

    print("\nDebug mode finished. Set DEBUG_SMALL = False for full training.\n")


if __name__ == "__main__":
    training_sequences_path = "data/CpG-islands.2K.seq.fa"
    training_labels_path = "data/CpG-islands.2K.lbl.fa"

    full_data = prepare_training_data(training_sequences_path, training_labels_path)
    train_data, dev_data, test_data = _split_data(full_data, train_frac=0.7, dev_frac=0.15, seed=999)

    if RUN_COMPARISON_PLOTS:
        os.makedirs("plots", exist_ok=True)
        run_dev_comparison_plots(train_data, dev_data)
        sys.exit(0)

    if DEBUG_SMALL and MODEL_TYPE == "markov_llr":
        os.makedirs("plots", exist_ok=True)
        debugging_and_dev(train_data, dev_data)
        sys.exit(0)

    train_aug = _augment_with_reverse_complement(train_data)
    print(len(train_aug), len(dev_data), len(test_data))
    # Baseline on dev set
    _majority_baseline(dev_data, name="Dev")

    # Train on train_data only
    classifier = train_classifier(train_data)

    print(f"\nUsing MODEL_TYPE = {MODEL_TYPE}\n")

    print("Train accuracy:")
    _evaluate_on_pairs(classifier, train_aug)

    print("Dev accuracy:")
    _evaluate_on_pairs(classifier, dev_data)

    print("Test accuracy:")
    _evaluate_on_pairs(classifier, test_data)
