import numpy as np

def fold_lightcurve(time, flux, period, t0=None):
    """
    Διπλώνει (phase-fold) την καμπύλη φωτός στην περιοδική περίοδο.
    Επιστρέφει phase, flux_sorted.
    """
    if t0 is None:
        t0 = time[0]  # αφετηρία φάσης
    phase = ((time - t0) % period) / period
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]