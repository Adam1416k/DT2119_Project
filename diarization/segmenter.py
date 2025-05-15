from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
import torchaudio
import os, sys
import tqdm
import torch

from diarization.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "voxconverse_test_wav")


def process_audio():
    # Step 1: Load segmentation model
    seg_model = Model.from_pretrained(
        "pyannote/segmentation-3.0", use_auth_token=HF_TOKEN
    ).to(device)
    vad_pipeline = VoiceActivityDetection(segmentation=seg_model).to(device)
    vad_pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    # Step 2: Load embedding model
    embed_model = Inference("pyannote/embedding", use_auth_token=HF_TOKEN).to(device)

    all_segments = []

    pbar = tqdm.tqdm(
        os.listdir(AUDIO_DIR),
        total=len(os.listdir(AUDIO_DIR)),
        desc=f"Segmenting Audio Files",
    )

    # Step 3: Loop through all wav files in the folder
    for filename in pbar:
        if not filename.endswith(".wav"):
            continue

        item = {}
        item["filename"] = filename

        audio_path = os.path.join(AUDIO_DIR, filename)

        try:
            # Load the audio file using torchaudio to get its duration
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_duration = waveform.size(1) / sample_rate  # duration in seconds

            # Run VAD to detect speech segments
            speech_regions = vad_pipeline(
                audio_path
            )  # returns pyannote.core.Annotation

            # Extract embeddings for each speech segment
            segments = []
            for segment in speech_regions.itersegments():
                # Ensure the segment is within the bounds of the audio file
                if segment.end <= audio_duration:
                    emb = embed_model.crop(audio_path, segment)
                    segments.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "embedding": emb.data.tolist(),
                        }
                    )
                else:
                    pass
                    # print(f"Skipping out-of-bounds segment in {filename}: {segment}")

            # print(f"Segments for {filename} collected.")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

        item["segments"] = segments
        all_segments.append(item)

    return all_segments
