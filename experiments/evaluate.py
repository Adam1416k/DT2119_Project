import os, sys
import tqdm
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from diarization.evaluator import *

column_names = ["file_id", "DER", "FA", "Miss", "Conf"]

data_lists = [[] for _ in range(4)]

results_dirs = [
    os.listdir(os.path.join(PROJECT_ROOT, "results", method))
    for method in ["ahc", "bhmm", "spectral", "pyannote"]
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
            ["ahc", "bhmm", "spectral", "pyannote"],
            [ahc_file, bhmm_file, spectral_file, pyannote_file],
        )
    ]
    ref_path = os.path.join(PROJECT_ROOT, "data", "test", ref_file)

    for idx, data_list in enumerate(data_lists):
        try:
            data_list.append(load_and_compute_metrics(ref_path, hyp_paths[idx]))

        except IndexError:
            # Will raise an index error if the output file is empty.
            pass

means = []

for data_list, method in zip(data_lists, ["ahc", "bhmm", "spectral", "pyannote"]):
    df = pd.DataFrame(data_list, columns=column_names)
    path = os.path.join(PROJECT_ROOT, "results", method)
    df.to_csv(path + ".csv", sep="\t")
    means.append(df.mean(numeric_only=True))

means_df = pd.DataFrame(means, index=["ahc", "bhmm", "spectral", "pyannote"])
means_df.to_csv(os.path.join(PROJECT_ROOT, "results", "mean_results.csv"), sep="\t")
