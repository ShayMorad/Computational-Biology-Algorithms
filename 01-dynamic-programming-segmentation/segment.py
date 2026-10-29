import numpy as np
import argparse


def read_signal_file(filepath: str) -> np.ndarray:
    """
    Read signal data from a tab-separated text file.
    Each line contains one or more columns of values separated by tabs.

    Args:
        filepath: Path to the input text file

    Returns:
        np.ndarray: Array of signal values from all columns
    """
    return np.loadtxt(filepath, delimiter='\t', dtype=np.float32)


def print_segments(s: np.ndarray, c: float) -> None:
    """
    Print segments array with 3 decimal places.
    Each row contains [start, end, average].

    Args:
        s: numpy array of shape (n_segments, 3)
        c: Total cost value of optimal segmentation
    """
    for row in s:
        print(f"{int(row[0])} {int(row[1])} {np.round(row[2], 3)}")
    print(np.round(c, 3))
    return


def _prepare_prefix(x: np.ndarray):
    # build cumulative sums and squared sums for prefixes
    cum_sum = np.zeros_like(x, dtype=np.float64)
    cum_sq_sum = np.zeros_like(x, dtype=np.float64)
    cum_sum[1:] = np.cumsum(x[1:], axis=0)
    cum_sq_sum[1:] = np.cumsum(x[1:] * x[1:], axis=0)
    return cum_sum, cum_sq_sum


def _calc_seg_err(cum_sum, cum_sq_sum, start_pos, end_pos):
    seg_len = end_pos - start_pos
    s_vals = cum_sum[end_pos] - cum_sum[start_pos]
    s_sq_vals = cum_sq_sum[end_pos] - cum_sq_sum[start_pos]
    err = s_sq_vals - (s_vals ** 2) / seg_len
    return np.sum(err) if err.ndim > 0 else err


def _dp_calculations(x: np.ndarray, p: float, q: int):
    n = x.shape[0] - 1
    # create arrays fitting for the suggest score 'c' and traceback 't' arrays
    best_score = np.full(n + 1, np.inf, dtype=np.float64)
    best_score[0] = 0.0  # empty segment score
    back_ptr = np.zeros(n + 1, dtype=np.int32)

    prefix_sum, prefix_squared_sum = _prepare_prefix(x)  # using prefixes helps us calculate the actual segments cost, etc... fast

    for end_pos in range(1, n + 1):
        min_cost = np.inf
        split_at = 0
        for start_pos in range(max(0, end_pos - q), end_pos):
            seg_err = _calc_seg_err(prefix_sum, prefix_squared_sum, start_pos, end_pos)
            penalty_term = 0.0 if start_pos == 0 else p
            cost_now = best_score[start_pos] + seg_err + penalty_term
            if cost_now < min_cost:
                min_cost = cost_now
                split_at = start_pos
        best_score[end_pos] = min_cost
        back_ptr[end_pos] = split_at
    return best_score, back_ptr


def segment(x: np.ndarray, p: float, q: int):
    """
    Segment a signal using dynamic programming.

    Args:
        x: Input signal (numpy 1d array)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 3), each row in format [start, end, average]
        c: Total cost value of optimal segmentation
    """
    n = len(x)
    # add empty point the empty prefix so that DP works as in the usual conventions we know
    x2d = np.zeros((n + 1,), dtype=np.float64)
    x2d[1:] = x
    best_score, back_ptr = _dp_calculations(x2d, p, q)  # best_score = "c", back_ptr = "t" arrays suggested in exercise

    # traceback
    cum_sum = np.concatenate(([0.0], np.cumsum(x)))
    segments = []
    idx = n
    while idx > 0:
        j = back_ptr[idx]
        seg_start = j + 1  # add one because we have the "empty" first segment for the base case as in normal DP..
        seg_end = idx
        avg_val = (cum_sum[idx] - cum_sum[j]) / (idx - j)
        segments.append([seg_start, seg_end, avg_val])
        idx = j
    segments.reverse()
    return np.array(segments, dtype=float), float(best_score[n])


def segment_multi_channel(x: np.ndarray, p: float, q: int):
    """
    Similar to segment but with multiple channels.

    Args:
        x: Input signal (numpy 2d array, where rows are different channels)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 2), each row in format [start, end]
        c: Total cost value of optimal segmentation
    """

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    x2d = np.zeros((n + 1, x.shape[1]), dtype=np.float64)
    x2d[1:] = x
    best_score, back_ptr = _dp_calculations(x2d, p, q)

    # traceback
    cum_sum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    segments = []
    idx = n
    while idx > 0:
        j = back_ptr[idx]
        seg_start = j + 1
        seg_end = idx
        seg_means = (cum_sum[idx] - cum_sum[j]) / (idx - j)
        avg_val = np.mean(seg_means)  # average over all d channels
        segments.append([seg_start, seg_end, avg_val])
        idx = j
    segments.reverse()
    return np.array(segments, dtype=float), float(best_score[n])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process signal data from a text file.')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--penalty', type=float, required=True, help='Penalty parameter')
    parser.add_argument('--max_len', type=int, required=True, help='Maximum segment length')
    parser.add_argument('--is_multi_channel', type=bool, required=False, default=False,
                        help='call segment_multi_channel')
    args = parser.parse_args()
    seq = read_signal_file(args.filepath)
    segments, cost = None, None

    if args.is_multi_channel:
        segments, cost = segment_multi_channel(seq, args.penalty, args.max_len)
    else:
        segments, cost = segment(seq, args.penalty, args.max_len)
    print_segments(segments, cost)
