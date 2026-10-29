import numpy as np
import matplotlib.pyplot as plt
import time
from segment import segment


# -------------------------------------------------------------------
# Synthetic data generator
# -------------------------------------------------------------------
def generate_signal(n, noise=0.2):
    """Generate a synthetic piecewise-constant 1D signal."""
    x = np.zeros(n)
    step_points = sorted(np.random.choice(np.arange(10, n-10), size=5, replace=False))
    current_val = 0
    last_idx = 0
    for sp in step_points + [n]:
        current_val += np.random.uniform(-2, 2)
        x[last_idx:sp] = current_val
        last_idx = sp
    x += np.random.normal(0, noise, n)
    return x


# -------------------------------------------------------------------
# Runtime experiments
# -------------------------------------------------------------------
def measure_runtime_vs_n(n_values, q=50, p=0.1):
    times = []
    for n in n_values:
        x = generate_signal(n)
        start = time.perf_counter()
        segment(x, p, q)
        times.append(time.perf_counter() - start)

    plt.figure(figsize=(7, 5))
    plt.plot(n_values, times, 'o-', color='royalblue', label=f"q={q}")
    plt.xlabel("Sequence length n")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs n (fixed q)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def measure_runtime_vs_q(q_values, n=1000, p=0.1):
    times = []
    x = generate_signal(n)
    for q in q_values:
        start = time.perf_counter()
        segment(x, p, q)
        times.append(time.perf_counter() - start)

    plt.figure(figsize=(7, 5))
    plt.plot(q_values, times, 'o-', color='darkorange', label=f"n={n}")
    plt.xlabel("Maximum segment length q")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs q (fixed n)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def measure_runtime_n_q_grid(n_values, q_values, p=0.1):
    """Measure runtime as a 2D grid (for n × q) and visualize as heatmap."""
    times = np.zeros((len(n_values), len(q_values)))
    for i, n in enumerate(n_values):
        x = generate_signal(n)
        for j, q in enumerate(q_values):
            start = time.perf_counter()
            segment(x, p, q)
            times[i, j] = time.perf_counter() - start
            print(f"n={n}, q={q} -> {times[i, j]:.4f}s")

    # Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(times, aspect='auto', origin='lower',
               extent=[min(q_values), max(q_values), min(n_values), max(n_values)],
               cmap='viridis')
    plt.colorbar(label='Runtime (seconds)')
    plt.xlabel("Maximum segment length q")
    plt.ylabel("Sequence length n")
    plt.title("Runtime heatmap (n vs q)")
    plt.tight_layout()
    plt.show()

    return times


def visualize_segmentation(x, segments, penalty, max_len):
    """Visualize a segmentation result (like the example plot)."""
    plt.figure(figsize=(10, 2.5))
    plt.scatter(np.arange(len(x)), x, s=8, color='blue', label='Signal')

    colors = plt.cm.tab10(np.arange(len(segments)) % 10)
    for idx, (start, end, avg) in enumerate(segments):
        plt.hlines(avg, start-1, end-1, colors=colors[idx], lw=4)

    plt.title(f"penalty = {penalty}, max_len = {max_len}")
    plt.xlabel("Index")
    plt.ylabel("value")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Run all analyses
# -------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Parameter sets for runtime analysis
    n_values = [200, 400, 800, 1600, 3200]
    q_values = [10, 50, 100, 200, 400]

    # --- Runtime measurements ---
    measure_runtime_vs_n(n_values, q=100)
    measure_runtime_vs_q(q_values, n=1000)
    measure_runtime_n_q_grid(n_values, q_values, p=0.1)

    # --- Example segmentation visualization ---
    print("\nVisualizing example segmentation...\n")
    n = 350
    x = generate_signal(n)
    penalty = [0.1, 0.5]
    max_len = 50
    segments, cost = segment(x, penalty[0], max_len)
    visualize_segmentation(x, segments, penalty[0], max_len)
    segments, cost = segment(x, penalty[1], max_len)
    visualize_segmentation(x, segments, penalty[1], max_len)