import math


def concentration_score(counts: dict[str, int]) -> float:
    """
    1 - all abstracts in the cluster are concentrated in a single arXiv category
    0 - the categories within the cluster are perfectly uniform (i.e., completely mixed)
    """
    values = list(counts.values())
    K = len(values)

    if K == 1:
        return 1.0

    total = sum(values)
    if total == 0:
        return 0.0

    ps = [v / total for v in values if v > 0]

    H = -sum(p * math.log(p) for p in ps)
    H_norm = H / math.log(K)

    return 1.0 - H_norm
