import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diarization.clustering.ahc import process_and_cluster_audio


# Save RTTM files function
def save_rttm(file_id, segments, out_file):
    # Remove the .wav extension from the filename

    with open(out_file, "w") as f:
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            cluster = segment["cluster"]
            cluster_id = f"spk{cluster:02d}"  # Convert cluster to spkXX format
            duration = end - start
            # Write with start and duration formatted to 5 decimal places
            f.write(
                f"SPEAKER {file_id} 1 {start:.5f} {duration:.5f} <NA> <NA> {cluster_id} <NA> <NA>\n"
            )
    # print(f"RTTM file saved for {filename_without_extension} at {rttm_filename}")


def run_ahc(all_data):
    # Save RTTM Files
    results_dir = os.path.join(PROJECT_ROOT, "results", "ahc")
    os.makedirs(results_dir, exist_ok=True)

    all_segments = process_and_cluster_audio(all_data)
    for audio_data in all_segments:
        filename = audio_data["filename"]
        segments = audio_data["segments"]

        file_id = os.path.splitext(filename)[0]
        out_file = os.path.join(results_dir, f"{file_id}.rttm")

        save_rttm(filename, segments, out_file)
