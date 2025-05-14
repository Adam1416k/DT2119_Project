import sys
import os
import numpy as np
sys.path.append('/Users/mythernstrom/Documents/GitHub/DT2119_Project/diarization/clustering')
from ahc import process_and_cluster_audio  

# Paths
AUDIO_DIR = "./data/audio/voxconverse_test_wav 4"
OUTPUT_DIR = "./ahc_segments"
RTTM_OUTPUT_DIR = "./rttm_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure the RTTM directory exists
print(f"Ensuring RTTM directory exists at {RTTM_OUTPUT_DIR}")
os.makedirs(RTTM_OUTPUT_DIR, exist_ok=True)  # Ensure the RTTM output directory exists

# Save RTTM files function
def save_rttm(filename, segments, output_dir):
    # Remove the .wav extension from the filename
    filename_without_extension = os.path.splitext(filename)[0]
    rttm_filename = os.path.join(output_dir, f"{filename_without_extension}.rttm")
    
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Saving RTTM file to: {rttm_filename}")
    
    with open(rttm_filename, 'w') as f:
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            cluster = segment["cluster"]
            cluster_id = f"spk{cluster:02d}"  # Convert cluster to spkXX format
            duration = end - start
            # Write with start and duration formatted to 5 decimal places
            f.write(f"SPEAKER {filename_without_extension} 1 {start:.5f} {duration:.5f} <NA> <NA> {cluster_id} <NA> <NA>\n")
    print(f"RTTM file saved for {filename_without_extension} at {rttm_filename}")

# Process and cluster audio data
all_segments = process_and_cluster_audio()

# Save results as RTTM files
for audio_data in all_segments:
    filename = audio_data["filename"]
    segments = audio_data["segments"]

    # Save the clustering result as an RTTM file
    save_rttm(filename, segments, RTTM_OUTPUT_DIR)

print("AHC clustering and RTTM generation complete.")
