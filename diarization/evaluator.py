from pyannote.database.util import load_rttm
from pyannote.core import Annotation

# Suppress UserWarning: 'uem' was approximated by the union of 'reference' and 'hypothesis' extents.
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.metrics.utils")


def load_and_compute_metrics(ref_path, hyp_path, metric):
    """
    Load the requested RTTM files and computes the metrics.
    """

    ref_rttm = load_rttm(ref_path)
    hyp_rttm = load_rttm(hyp_path)

    # If the rttm file is empty, load_rttm will return an empty dictionary
    # If that is the case create an empty instance to get the correct evaluation
    if len(hyp_rttm) == 0:
        hyp_rttm = Annotation()

    else:
        # If the rttm file is not empty, if will return a dictionary where the first
        # key will contain the pyannote annotation for the file
        hyp_rttm = hyp_rttm[list(hyp_rttm.keys())[0]]

    ref_rttm = ref_rttm[list(ref_rttm.keys())[0]]

    return metric(ref_rttm, hyp_rttm)
