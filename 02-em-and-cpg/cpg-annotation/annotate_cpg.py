from Bio import SeqIO  # pip install biopython
import argparse
import gzip
from development_cpg import (
    _augment_with_reverse_complement,
    _train_hmm_2state,
    _train_markov_llr,
    _evaluate_on_pairs,
    _viterbi_decode,
    _annotate_with_markov_llr,
)

MODEL_TYPE = "markov_llr"
LLR_WINDOW = 175

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

# =====================================================
#                   MODEL TRAINING
# =====================================================

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


def annotate_fasta_file(model, input_path, output_path):
    """
    Annotates all sequences in a FASTA file with CpG island predictions.

    Parameters:
        model (object): A trained classifier model.
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file where annotations will be saved.

    Writes:
        A gzipped FASTA file containing predicted annotations for each input sequence.
    """
    sequences = parse_fasta_file(input_path)

    with gzip.open(output_path, 'wt') as gzipped_file:
        for seq_id, sequence in sequences.items():
            annotation = annotate_sequence(model, sequence)
            gzipped_file.write(f">{seq_id}\n{annotation}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict CpG islands in DNA sequences.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file containing DNA sequences.")
    parser.add_argument("--output_file", type=str, help="Path to the output FASTA file for saving predictions.")
    # optional args I added for being able to choose models through CLI
    parser.add_argument("--model_type", type=str, default="markov_llr", choices=["hmm_2state", "markov_llr"], help="Which model to train.")
    parser.add_argument("--llr_window", type=int, default=175,          help="Window size for Markov LLR model.")

    args = parser.parse_args()

    training_sequences_path = r"data/CpG-islands.2K.seq.fa"
    training_labels_path = r"data/CpG-islands.2K.lbl.fa"

    # ======================== START MODEL TRAINING & EVALUATION ======================== #
    MODEL_TYPE = args.model_type
    LLR_WINDOW = args.llr_window

    # Prepare labeled data (full set)
    train_data = prepare_training_data(training_sequences_path, training_labels_path)

    # Augment train data with reverse complement:
    train_data = _augment_with_reverse_complement(train_data)

    # Train
    classifier = train_classifier(train_data)

    print(f"\nUsing MODEL_TYPE = {MODEL_TYPE}\n")

    print("Train accuracy:")
    _evaluate_on_pairs(classifier, train_data)

    # ======================== END MODEL TRAINING & EVALUATION ======================== #

    # Annotate sequences and save predictions
    if args.fasta_path is not None and args.output_file is not None:
        annotate_fasta_file(classifier, args.fasta_path, args.output_file)
    else:
        print("No --fasta_path / --output_file given, skipping annotation export.")

