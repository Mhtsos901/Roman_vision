import lightkurve as lk
import numpy as np
import os


class DataLoader:
    def __init__(self, cache_dir='./data/raw'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_single_tic(self, tic_id, mission='TESS', sector=None, author='SPOC'):
        search_result = lk.search_lightcurve(
            f"TIC {tic_id}", mission=mission, author=author, sector=sector
        )
        if len(search_result) == 0:
            raise ValueError(f"No light curve found for TIC {tic_id}")

        lc_collection = search_result.download_all(download_dir=self.cache_dir)
        if isinstance(lc_collection, lk.LightCurveCollection):
            lc = lc_collection[0]
        else:
            lc = lc_collection
        return lc

    def lightcurve_to_arrays(self, lc):
        time = lc.time.value
        flux = lc.flux.value
        if hasattr(lc, 'flux_err') and lc.flux_err is not None:
            flux_err = lc.flux_err.value
        else:
            flux_err = np.sqrt(np.abs(flux) + 1e-10)
        return time, flux, flux_err