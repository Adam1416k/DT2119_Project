from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

# Suppress UserWarning: 'uem' was approximated by the union of 'reference' and 'hypothesis' extents.
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.metrics.utils")

metric = DiarizationErrorRate()


def compute_metrics(reference, hypothesis):
    """
    Calculates the metrics DER%, FA%, Miss%, Conf% for reference and hypothesis.
    """

    components = metric(reference, hypothesis, detailed=True)

    DER = components["diarization error rate"]
    FA = components["false alarm"]
    Miss = components["missed detection"]
    Conf = components["confusion"]

    return DER, FA, Miss, Conf


def load_and_compute_metrics(ref_path, hyp_path):
    """
    Load the requested RTTM files and computes the metrics.
    """

    ref_rttm = load_rttm(ref_path)
    hyp_rttm = load_rttm(hyp_path)

    return list(ref_rttm.keys())[0], *compute_metrics(
        ref_rttm[list(ref_rttm.keys())[0]], hyp_rttm[list(hyp_rttm.keys())[0]]
    )
