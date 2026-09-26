"""Small, deterministic Pareto archive utilities for Q3's minimization goals."""


def dominates(left, right, tolerance=1e-9):
    """Return True when objective vector ``left`` weakly dominates ``right``."""
    if len(left) != len(right):
        raise ValueError("Objective vectors must have the same dimension")
    return (all(float(a) <= float(b) + tolerance for a, b in zip(left, right))
            and any(float(a) < float(b) - tolerance for a, b in zip(left, right)))


def update_archive(archive, candidate, objective_key="objectives", tolerance=1e-9):
    """Insert a feasible candidate and return a new nondominated archive.

    ``candidate[objective_key]`` may be either a mapping or a sequence. Exact
    objective duplicates are retained only once, with the existing entry kept.
    """
    def vector(item):
        values = item[objective_key]
        return tuple(float(value) for value in values.values()) if isinstance(values, dict) else tuple(map(float, values))

    target = vector(candidate)
    current = list(archive)
    for item in current:
        value = vector(item)
        if (all(abs(a - b) <= tolerance for a, b in zip(value, target))
                or dominates(value, target, tolerance)):
            return current
    return [item for item in current if not dominates(target, vector(item), tolerance)] + [candidate]
