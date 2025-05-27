import os, sys
import tqdm
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from pyannote.metrics.diarization import DiarizationErrorRate
from diarization.evaluator import *

methods = ["ahc", "bhmm", "spectral", "pyannote"]
column_names = ["file_id", "DER", "FA", "Miss", "Conf"]

metrics = [DiarizationErrorRate() for _ in range(len(methods))]

results_dirs = [
    os.listdir(os.path.join(PROJECT_ROOT, "results", method)) for method in methods
]
reference_dirs = os.listdir(os.path.join(PROJECT_ROOT, "data", "test"))

pbar = tqdm.tqdm(
    zip(reference_dirs, *results_dirs),
    total=len(reference_dirs),
    desc=f"Calculating metrics",
)

for ref_file, ahc_file, bhmm_file, spectral_file, pyannote_file in pbar:
    # Check for conistency
    if not ref_file == ahc_file == bhmm_file == spectral_file == pyannote_file:
        raise "Tried to access different files"

    hyp_paths = [
        os.path.join(PROJECT_ROOT, "results", method, file_id)
        for method, file_id in zip(
            methods,
            [ahc_file, bhmm_file, spectral_file, pyannote_file],
        )
    ]
    ref_path = os.path.join(PROJECT_ROOT, "data", "test", ref_file)

    for idx, metric in enumerate(metrics):
        load_and_compute_metrics(ref_path, hyp_paths[idx], metric)


means = []

for metric, method in zip(metrics, methods):
    df = metric.report()
    path = os.path.join(PROJECT_ROOT, "results", method)
    df.to_csv(path + ".csv")
    means.append(df.loc["TOTAL"])

means = pd.DataFrame(means, index=methods)
path = os.path.join(PROJECT_ROOT, "results", "mean_results.csv")
means.to_csv(path)
