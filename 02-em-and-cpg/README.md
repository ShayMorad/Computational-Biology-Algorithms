# Expectation-maximization and CpG sequence annotation

Two related probabilistic sequence-analysis implementations: EM training on a coin-sequence HMM and CpG-island prediction from labeled nucleotide data.

## Hidden Markov model and EM

[`coin-hmm/q_em.py`](coin-hmm/q_em.py) implements a two-state hidden Markov model, experiments with parameter initializations, and tracks training log-likelihood.

![EM log-likelihood by iteration](results/coins_loglik_0.png)

## CpG annotation

[`cpg-annotation/annotate_cpg.py`](cpg-annotation/annotate_cpg.py) provides the annotation entry point. [`development_cpg.py`](cpg-annotation/development_cpg.py) contains supporting model fitting, window-size analysis, and evaluation code, including HMM and Markov log-likelihood-ratio approaches.

![CpG model accuracy comparison](results/dev_accuracy_per_model.png)

[Additional CpG composition experiment](results/true_vs_pred_c_fraction.png) · [Window-size analysis](results/window_vs_accuracy_debug.png)

## Input requirements

- **Coin HMM:** a FASTA sequence such as `data/q_coins.seq.fa` relative to `coin-hmm/`.
- **CpG training:** paired sequence/label FASTA inputs such as `data/CpG-islands.2K.seq.fa` and `data/CpG-islands.2K.lbl.fa` under `cpg-annotation/`.
- **Prediction export:** `annotate_cpg.py` accepts `--fasta_path` and `--output_file`.

Input FASTA datasets and any fitted model artifacts are external to the repository. Results shown above reflect the recorded experiment configuration.

**Skills:** expectation-maximization, HMMs, sequence classification, probabilistic feature design, model evaluation.
