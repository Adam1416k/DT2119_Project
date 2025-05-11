from pyannote.audio import Model, Inference, Pipeline
from pyannote.core import Annotation, Segment
import os
import pickle
import json
from data_loader import load_rttm_files

AUDIO_DIR = "./data/audio/voxconverse_test_wav 4"

def save_segments(segments, base_filename="segments"):
    # Save to JSON
    with open(f"{base_filename}.json", "w") as f_json:
        json.dump(segments, f_json, indent=2)

    # Save to Pickle
    with open(f"{base_filename}.pkl", "wb") as f_pickle:
        pickle.dump(segments, f_pickle)

def segment_audio(inference: Inference, file_path: str):
    """
    Perform segmentation using the provided inference object.
    """
    return inference(file_path)

def extract_segments(diarization: Annotation):
    """
    Convert pyannote Annotation (diarization) to a list of dictionaries.
    """
    segments = []
    if isinstance(diarization, Annotation):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': float(turn.start),
                'end': float(turn.end),
                'duration': float(turn.duration),
                'speaker': str(speaker)
            })
    else:
        print("Diarization is not in expected format or is empty.")

    save_segments(segments)
    return segments

if __name__ == "__main__":
    # Initialize the pipeline with the correct model
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", 
        use_auth_token="your_auth_token"
    )

    # List of audio files to process
    audio_files = [
        "./data/audio/voxconverse_test_wav 4/aepyx.wav"
    ]

    for audio_file in audio_files:
        print(f"Running diarization on: {audio_file}")

        # Perform diarization using the pipeline
        diarization = pipeline(audio_file)
        print("Diarization complete:", diarization)

        # Extract segments from the diarization
        segments = extract_segments(diarization)

        # Display first 5 segments
        print(f"First 5 segments for {audio_file}:")
        for s in segments[:5]:
            print(s)