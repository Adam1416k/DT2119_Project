from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
import torchaudio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("token.env")

# Get the token securely from the environment
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN environment variable.")

# Paths
AUDIO_DIR = "./data/audio/voxconverse_test_wav 4"

def process_audio():
    # Step 1: Load segmentation model
    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)
    vad_pipeline = VoiceActivityDetection(segmentation=seg_model)
    vad_pipeline.instantiate({
        "min_duration_on": 0.0,
        "min_duration_off": 0.0
    })

    # Step 2: Load embedding model
    embed_model = Inference("pyannote/embedding", use_auth_token=HF_TOKEN)

    all_segments = []

    # Step 3: Loop through all wav files in the folder
    for filename in os.listdir(AUDIO_DIR):
        if not filename.endswith(".wav"):
            continue

        audio_path = os.path.join(AUDIO_DIR, filename)
        print(f"Processing {filename}...")

        try:
            # Load the audio file using torchaudio to get its duration
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_duration = waveform.size(1) / sample_rate  # duration in seconds

            # Run VAD to detect speech segments
            speech_regions = vad_pipeline(audio_path)  # returns pyannote.core.Annotation

            # Extract embeddings for each speech segment
            segments = []
            for segment in speech_regions.itersegments():
                # Ensure the segment is within the bounds of the audio file
                if segment.end <= audio_duration:
                    emb = embed_model.crop(audio_path, segment)
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "embedding": emb.data.tolist()
                    })
                else:
                    print(f"Skipping out-of-bounds segment in {filename}: {segment}")

            print(f"Segments for {filename} collected.")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    
    return all_segments

if __name__ == "__main__":
    all_segments = process_audio()
    print("Segmentation complete.")