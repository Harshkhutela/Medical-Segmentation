"""Selection strategies for choosing images to annotate."""


def select_top_k(scores, k):
    """Return the ``k`` highest-scoring uncertain image records.

    Args:
        scores: Iterable of dictionaries containing a ``final_score`` key.
        k: Maximum number of images to select.

    Returns:
        A descending list of the most uncertain image records.
    """
    if k <= 0:
        return []

    sorted_scores = sorted(
        scores,
        key=lambda record: record["final_score"],
        reverse=True,
    )
    return sorted_scores[:k]
