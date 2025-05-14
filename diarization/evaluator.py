from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

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

    # Should only be one key that is the same for both files, and the key should be the file id
    file_id = list(ref_rttm.keys())[0]
    return compute_metrics(ref_rttm[file_id], hyp_rttm[file_id])
