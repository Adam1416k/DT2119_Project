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
