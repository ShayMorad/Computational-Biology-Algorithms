import os
import numpy as np
import matplotlib.pyplot as plt

P0_candidates = [
    [0.5, 0.5],  # symmetric
    [0.9, 0.1],  # start mostly in state 0
    [0.1, 0.9],  # start mostly in state 1
]
P0_random = np.random.rand(2).tolist()  # 1D array -> Python list [x, y]
P0_candidates.insert(0, P0_random)

# =====================================================
#                 FASTA SEQUENCE READER
# =====================================================
def read_fasta_sequences(path):
    """Reads FASTA sequences into a list of strings."""
    seqs = []
    with open(path) as f:
        current = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            seqs.append("".join(current))
    return seqs

# =====================================================
#            LOG-LIKELIHOOD PLOTTING HELPER
# =====================================================
def plot_loglik_histories(all_runs, filename_prefix="loglik_run"):
    """
    For each EM run in all_runs, save a plot of
        iteration vs log P(X | θ^{(j)}).

    all_runs is the list returned by run_multiple_inits.
    """

    os.makedirs("plots", exist_ok=True)
    for idx, run in enumerate(all_runs):
        history = run["loglik_history"]
        iters = np.arange(1, len(history) + 1)

        plt.figure()
        plt.plot(iters, history)
        plt.xlabel("EM iteration")
        plt.ylabel("log P(X | θ^(j))")
        plt.title(f"Run {idx}  (Initialized P0 = {run['Initialized P0']})")
        plt.tight_layout()

        filename = f"{filename_prefix}_{idx}.png"
        filepath = os.path.join("plots", filename)
        plt.savefig(filepath)
        plt.close()

# =====================================================
#                   HMM_EM CLASS
# =====================================================
class HMM_EM:
    """
    Simple HMM for the part 1 of the exercise, 'coins', using the same notation as in the lecture:
    - Hidden states: H_i  ∈ {F, L}  (self.states)
    - Observations: X_i  ∈ {H, T}  (self.observations)

    Parameters (θ):
    - P0[s]          initial distribution P(H_1 = s)
    - τ[t,s]         transition P(H_i = s | H_{i-1} = t)
                     (row = 'from' state t, column = 'to' state s)
    - π[s,x]         emission P(X_i = x | H_i = s)
                     (row = state s, column = observation symbol x)

    Forward matrix F:
        F[i,s] = P(H_i = s, x_1, ..., x_i | θ)
    Backward matrix B:
        B[i,s] = P(x_{i+1}, ..., x_n | H_i = s, θ)

    Important note about indices:
    ------------------------------
    In the lecture, time indices start at 1:  i = 1, 2, ..., n.
    In Python, array indices start at 0:      t = 0, 1, ..., T-1.

    We keep the mathematical meaning the same, we just shift by 1:

        i (lecture) = t + 1  (code)

    So:
        F[t, s] in the code  ==  F_i[s] in the slides, with i = t+1
        B[t, s] in the code  ==  B_i[s] in the slides, with i = t+1
    """

    def __init__(self, states, observations):
        self.last_loglik = None
        self.loglik_history = []
        self.states = states  # all possible H values
        self.observations = observations  # all possible X values

        # Number of states (H's) and observation symbols (X's)
        self.N = len(states)  # |H|
        self.M = len(observations)  # |X|

        # -------------------------------------------------------
        # INITIALIZATIONS (random τ, π as in exercise)
        # -------------------------------------------------------
        # Initial state probabilities P0[s] = P(H_1 = s)
        # Here we start with a symmetric prior (fair vs loaded).
        self.P0 = np.array([0.5, 0.5], dtype=float)

        # Transition matrix τ[s,t] = P(H_i = t | H_{i-1} = s)
        # self.tau[i, j] = P(H(t) = states[j] | H(t-1) = states[i])
        # row = "from" state s, column = "to" state t
        self.tau = np.random.rand(self.N, self.N)
        self.tau = self.tau / self.tau.sum(axis=1, keepdims=True)

        # Emission matrix π[s,x] = P(X_i = x | H_i = s)
        # self.pi[i, k] = P(X(t) = observations[k] | H(t) = states[i])
        # row = state s, column = observation symbol x (0 = 'H', 1 = 'T')
        self.pi = np.random.rand(self.N, self.M)
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

        # # we override the initial distributions so print outside instead and comment out these lines
        # print("Initial distribution P0[s]:\n", self.P0)
        # print("Transition matrix τ[s,t]:\n", self.tau)
        # print("Emission matrix π[s,x]:\n", self.pi)

        # Observation symbol → index lookup (e.g., 'H' -> 0, 'T' -> 1)
        self.obs_index = {o: i for i, o in enumerate(observations)}

    # =====================================================
    #                   HELPER FUNCTIONS
    # =====================================================
    def _encode(self, obs_seq):
        """
        Convert the sequence of observation symbols [x1,x2,...,xn] to an integer [0,1].
        """
        return np.array([self.obs_index[o] for o in obs_seq], dtype=int)

    @staticmethod
    def _log_sum_exp(log_values):
        """
        Compute log(sum_i exp(log_values[i]) ) in a numerically stable way:
            log ∑ e^{c_i} = max_i c_i + log ∑ e^{c_i - max_i c_i}
        """
        m = np.max(log_values)
        return m + np.log(np.sum(np.exp(log_values - m)))

    def set_P0(self, P0):
        self.P0 = P0

    def set_tau(self, tau):
        self.tau = tau

    def set_pi(self, pi):
        self.pi = pi

    # =====================================================
    #                   FORWARD ALGORITHM
    # =====================================================
    def forward(self, obs_seq):
        """
        Forward pass.

        Computes the forward matrix F as in the lecture:

            F_i[s] = P(Hi = s, x1, ..., xi | θ)

        In code: F[t, s] corresponds to F_i[s] with i = t+1. For example F[0,s] in code = F1[s] in lecture.

        Parameters
        ----------
        obs_seq : str
            Sequence x1...xm (e.g. 'HTHHHT...').

        Returns
        -------
        F : np.ndarray of shape (T, N)
            F[t, s] = Fi[s] with i = t+1.
        """

        X_encoded = self._encode(obs_seq)  # Encode whole sequence where Xi encoded as 0/1
        T = len(X_encoded)  # length of current sequence
        F = np.zeros((T, self.N), dtype=float)  # matrix F we calculate in the Forward algorithm

        # ----- Initialization for i = 1 (t = 0) -----
        # F1[s] = P0[s] * π[s, x1]
        for s in range(self.N):
            F[0, s] = self.P0[s] * self.pi[s, X_encoded[0]]

        # ----- Recursion for i = 2..T (t = 1..T-1) -----
        # Lecture formula:
        #   Fi[s] = π[xi, s] * sum_t F_{i-1}[t] * τ[t, s]
        for t in range(1, T):  # i = t+1
            x_t = X_encoded[t]  # xi
            for s in range(self.N):  # current state s
                sum_over_prev = 0.0
                for prev in range(self.N):  # previous state t
                    # F_{i-1}[t] * τ[t,s]
                    sum_over_prev += F[t - 1, prev] * self.tau[prev, s]
                # multiply by emission π[s, xi]
                F[t, s] = self.pi[s, x_t] * sum_over_prev

        return F

    # =====================================================
    #                   BACKWARD ALGORITHM
    # =====================================================
    def backward(self, obs_seq):
        """
        Backward pass.

        B_i[s] = P(x_{i+1}, ..., x_n | H_i = s, θ)

        In code: B[t, s] corresponds to B_i[s] with i = t+1 from lecture.
        """
        X_encoded = self._encode(obs_seq)
        T = len(X_encoded)
        B = np.zeros((T, self.N), dtype=float)

        # ----- initialization: i = n (t = T-1) -----
        for s in range(self.N):
            B[T - 1, s] = 1.0

        # ----- recursion: i = n-1..1 (t = T-2..0) -----
        # B_i[s] = Σ_t τ[s,t] * π[t, x_{i+1}] * B_{i+1}[t]
        for t in range(T - 2, -1, -1):
            x_next = X_encoded[t + 1]  # x_{i+1}
            for s in range(self.N):  # go over all states for "Bi"
                sum_over_next = 0.0
                for nxt in range(self.N):  # use "Bi+1" with all states and corresponding tau
                    sum_over_next += (self.tau[s, nxt] * self.pi[nxt, x_next] * B[t + 1, nxt])
                B[t, s] = sum_over_next

        return B

    # =====================================================
    #                   BAUM–WELCH / EM
    # =====================================================
    def baum_welch(self, seqs, max_iter=150):
        """
        Train the HMM using Baum–Welch EM, directly in terms of the
        expected counts from the lecture:

            E[N_{0,s} | X, θ̃]
            E[N_{s,x} | X, θ̃]
            E[N_{s,t} | X, θ̃]

        For each sequence X^(m):

            P(X^(m) | θ̃) = sum_s F_n^m[s] - seen in lecture, can compute as well with Backwards

            P(H_i[m] = s | X^(m), θ̃) = F_i^m[s] * B_i^m[s] / P(X^(m) | θ̃)

            P(H_i[m] = s, H_{i+1}[m] = t | X^(m), θ̃) = F_i^m[s] * B_{i+1}^m[t] * τ[s,t] * π[t, x_{i+1}^m] / P(X^(m) | θ̃)

        We sum these terms over all sequences and positions to get
        the expectations, and then do MLE updates.
        """
        self.loglik_history = []

        for it in range(max_iter):
            # Expected counts as in the slides:
            # N0[s]   ~ E[N_{0,s} | X, θ̃]
            # N_sx[s,x] ~ E[N_{s,x} | X, θ̃]
            # N_st[s,t] ~ E[N_{s,t} | X, θ̃]
            expected_N0 = np.zeros(self.N)
            expected_N_sx = np.zeros((self.N, self.M))
            expected_N_st = np.zeros((self.N, self.N))

            total_loglik = 0.0

            for seq in seqs:
                X_encoding = self._encode(seq)
                T = len(X_encoding)

                # Forward and Backward for this sequence
                F = self.forward(seq)
                B = self.backward(seq)

                # ----- P(X | θ̃) with log-sum-exp to be safe -----
                last_F = F[T - 1, :]
                logPX = self._log_sum_exp(np.log(last_F + 1e-300))
                PX = np.exp(logPX)
                total_loglik += logPX


                # ---------- E[N_{0,s} | X^(m), θ̃] ----------
                # = P(H_1[m] = s | X^(m), θ̃)
                for s in range(self.N):
                    num = F[0, s] * B[0, s]
                    expected_N0[s] += num / PX

                # ---------- E[N_{s,x} | X^(m), θ̃] ----------
                # sum over positions i where x_i = x:
                # P(H_i[m] = s | X^(m), θ̃)
                for i in range(T):
                    x = X_encoding[i]
                    for s in range(self.N):
                        num = F[i, s] * B[i, s]
                        expected_N_sx[s, x] += num / PX

                # ---------- E[N_{s,t} | X^(m), θ̃] ----------
                # sum over i = 1..n-1 of:
                # P(H_i[m] = s, H_{i+1}[m] = t | X^(m), θ̃)
                for i in range(T - 1):
                    x_next = X_encoding[i + 1]
                    for s in range(self.N):
                        for t_state in range(self.N):
                            num = (
                                    F[i, s]
                                    * self.tau[s, t_state]
                                    * self.pi[t_state, x_next]
                                    * B[i + 1, t_state]
                            )
                            expected_N_st[s, t_state] += num / PX

            self.loglik_history.append(total_loglik)

            # ================= M-STEP =================
            # Use the expected counts to compute MLEs for P0, τ, π

            # --- update P0[s] ---
            total_N0 = expected_N0.sum()
            if total_N0 > 0:
                for s in range(self.N):
                    self.P0[s] = expected_N0[s] / total_N0

            # --- update τ[s,t] ---
            for s in range(self.N):
                row_sum = expected_N_st[s, :].sum()
                if row_sum > 0:
                    for t_state in range(self.N):
                        self.tau[s, t_state] = expected_N_st[s, t_state] / row_sum

            # --- update π[s,x] ---
            for s in range(self.N):
                row_sum = expected_N_sx[s, :].sum()
                if row_sum > 0:
                    for x in range(self.M):
                        self.pi[s, x] = expected_N_sx[s, x] / row_sum

            print(f"Iteration {it + 1}: log P(X | θ) = {total_loglik:.4f}")

        self.last_loglik = self.loglik_history[-1]
        return self.tau, self.pi


def run_multiple_inits(seqs, P0_list, restarts_per_P0=5, max_iter=150):
    best_model = None
    best_loglik = -np.inf
    results = []

    for P0 in P0_list:
        P0 = np.array(P0, dtype=float)
        P0 = P0 / P0.sum()  # just in case it's not normalized as a probability

        for r in range(restarts_per_P0):
            # new random HMM
            hmm = HMM_EM(states=["Fair", "Loaded"], observations=["H", "T"])
            # override initial distribution
            hmm.set_P0(P0.copy())
            # Print initialized parameters (for debugging)
            print("Initial distribution P0[s]:\n", hmm.P0)
            print("Transition matrix τ[s,t]:\n", hmm.tau)
            print("Emission matrix π[s,x]:\n", hmm.pi)

            tau, pi = hmm.baum_welch(seqs, max_iter=max_iter)

            run_info = {
                "Initialized P0": P0.copy(),
                "Final P0": hmm.P0.copy(),
                "tau": tau.copy(),
                "pi": pi.copy(),
                "loglik": hmm.last_loglik,
                "loglik_history": hmm.loglik_history.copy(),
            }
            results.append(run_info)

            if hmm.last_loglik > best_loglik:
                best_loglik = hmm.last_loglik
                best_model = run_info

    return best_model, results


# =============================================================
#                   MAIN PROGRAM (Student Entry Point)
# =============================================================
if __name__ == "__main__":
    # load data
    sequences = read_fasta_sequences("data/q_coins.seq.fa")

    # check: overall H/T frequency in dataset
    num_H = sum(seq.count("H") for seq in sequences)
    num_T = sum(seq.count("T") for seq in sequences)
    print(f"Percent of 'H': {num_H / (num_H + num_T)}")
    print(f"Percent of 'T': {num_T / (num_H + num_T)}\n")

    try:
        best_model, all_runs = run_multiple_inits(sequences, P0_list=P0_candidates, restarts_per_P0=3, max_iter=150)

        print("\nBest log-likelihood:", best_model["loglik"])
        print("Best Initialized P0:", best_model["Initialized P0"])
        print("Best Learned P0:", best_model["Final P0"])
        print("Best Learned τ[t,s]:\n", best_model["tau"])
        print("Best Learned π[s,x]:\n", best_model["pi"])

        # create one graph per run
        plot_loglik_histories(all_runs, filename_prefix="coins_loglik")

    except NotImplementedError as e:
        print("Missing implementation:", e)
