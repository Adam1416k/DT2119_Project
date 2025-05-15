#!/usr/bin/env python3

import os
import sys
import tqdm

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diarization.clustering.bhmm import cluster_segments as cluster_segments_bhmm


def write_rttm(file_id, segments, out_path):
    """
    Write segments to an RTTM file.
    """
    with open(out_path, "w") as f:
        for seg in segments:
            start = seg["start"]
            duration = seg["end"] - seg["start"]
            speaker = seg["speaker"]
            # RTTM format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
            line = (
                f"SPEAKER {file_id} 1 "
                f"{start:.3f} {duration:.3f} "
                "<NA> <NA> speaker{speaker} <NA> <NA>\n"
            ).format(speaker=speaker)
            f.write(line)


def run_bhmm(all_data):
    # 1) Prepare output directory
    results_dir = os.path.join(PROJECT_ROOT, "results", "bhmm")
    os.makedirs(results_dir, exist_ok=True)

    # 2) Segment & embed all audio
    # all_data = process_audio()

    pbar = tqdm.tqdm(
        all_data,
        total=len(all_data),
        desc=f"Performing BHMM",
    )

    # 3) Cluster & write RTTM per file
    for item in pbar:
        filename = item["filename"]
        file_id = os.path.splitext(filename)[0]
        segments = item["segments"]

        # BHMM clustering (defaults to 2 speakers unless you pass n_states=...)
        clustered = cluster_segments_bhmm(segments)

        out_file = os.path.join(results_dir, f"{file_id}.rttm")
        write_rttm(file_id, clustered, out_file)
        # print(f"Wrote BHMM RTTM → {out_file}")
