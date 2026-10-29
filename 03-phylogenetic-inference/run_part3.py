import os
import subprocess
import matplotlib.pyplot as plt

lengths = [50, 100, 250, 500, 750, 1000, 1250, 1500]
algorithms = ["NJ", "UPGMA"]

results = {
    "NJ": {"likelihood": [], "accuracy": []},
    "UPGMA": {"likelihood": [], "accuracy": []}
}


def read_true_ancestor(length):
    filepath = f"data/ancestor_{length}.fasta"
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Join lines ignoring the >header
        return "".join([l.strip() for l in lines if not l.startswith(">")])


for length in lengths:
    true_seq = read_true_ancestor(length)
    descendants_file = f"data/descendants_{length}.fasta"

    for algo in algorithms:
        print(f"Evaluating {algo} for length {length}...")

        # 1. Get Likelihood
        cmd_lik = ["python", "ex3.py", "--fasta", descendants_file, "--part", "likelihood", "--tree_type", algo]
        res_lik = subprocess.run(cmd_lik, capture_output=True, text=True)
        # Parse the output to find the likelihood value
        for line in res_lik.stdout.split('\n'):
            if "Log-Likelihood:" in line:
                ll_val = float(line.split(":")[1].strip())
                results[algo]["likelihood"].append(ll_val)

        # 2. Get Reconstruction
        cmd_recon = ["python", "ex3.py", "--fasta", descendants_file, "--part", "reconstruction", "--tree_type", algo]
        res_recon = subprocess.run(cmd_recon, capture_output=True, text=True)
        # Parse the output to find the reconstructed sequence
        recon_seq = ""
        for line in res_recon.stdout.split('\n'):
            if "Reconstructed Root Sequence:" in line:
                recon_seq = line.split(":")[1].strip()

        # 3. Calculate Accuracy
        matches = sum(1 for a, b in zip(true_seq, recon_seq) if a == b)
        accuracy = (matches / length) * 100
        results[algo]["accuracy"].append(accuracy)

print("\n--- DONE! Generating Plots... ---")

# --- Plot 1: Accuracy vs Length ---
plt.figure(figsize=(10, 5))
plt.plot(lengths, results["NJ"]["accuracy"], marker='o', label='NJ Tree')
plt.plot(lengths, results["UPGMA"]["accuracy"], marker='s', label='UPGMA Tree')
plt.title('Ancestral Reconstruction Accuracy vs. Sequence Length')
plt.xlabel('Sequence Length (bp)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('reconstruction_accuracy.png')
plt.close()

# --- Plot 2: Log-Likelihood vs Length ---
plt.figure(figsize=(10, 5))
plt.plot(lengths, results["NJ"]["likelihood"], marker='o', label='NJ Tree')
plt.plot(lengths, results["UPGMA"]["likelihood"], marker='s', label='UPGMA Tree')
plt.title('Tree Log-Likelihood vs. Sequence Length')
plt.xlabel('Sequence Length (bp)')
plt.ylabel('Log-Likelihood')
plt.legend()
plt.grid(True)
plt.savefig('log_likelihood_comparison.png')
plt.close()

print("Saved 'reconstruction_accuracy.png' and 'log_likelihood_comparison.png'")