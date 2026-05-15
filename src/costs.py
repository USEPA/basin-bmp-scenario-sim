from .sampling import sample_from_stats

def compute_bmp_cost_usd(rng, bmp_type_cps, unit_row, quantity, logger):
    """
    quantity: number in units matching 'unit' (e.g., ha, m, project)
    """
    if unit_row is None:
        return 0.0
    stats = {k: unit_row[k] for k in unit_row.index if k in ("mean","sd","min","max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
    rate = sample_from_stats(rng, stats, kind=None, verbose_logger=logger)
    if rate < 0:
        raise ValueError("Negative cost-rate sampled")
    total = rate * quantity
    if total < 0:
        raise ValueError("Negative total cost computed")
    return float(total)
