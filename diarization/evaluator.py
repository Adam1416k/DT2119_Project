from pyannote.database.util import load_rttm

# Suppress UserWarning: 'uem' was approximated by the union of 'reference' and 'hypothesis' extents.
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.metrics.utils")


def load_and_compute_metrics(ref_path, hyp_path, metric):
    """
    Load the requested RTTM files and computes the metrics.
    """

    ref_rttm = load_rttm(ref_path)
    hyp_rttm = load_rttm(hyp_path)

    return metric(
        ref_rttm[list(ref_rttm.keys())[0]], hyp_rttm[list(hyp_rttm.keys())[0]]
    )
