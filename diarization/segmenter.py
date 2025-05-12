from pyannote.audio import Model, Inference
from pyannote.core import Segment
import torchaudio
import os
import pickle
import json

# Step 1: Load segmentation model
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="use_auth_token")

# Step 2: Run VAD
vad_pipeline = VoiceActivityDetection(segmentation=seg_model)
vad_pipeline.instantiate({
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
})
speech_regions = vad_pipeline("./data/audio/voxconverse_test_wav 4/aepyx.wav")  # returns pyannote.core.Annotation

# Step 3: Load embedding model
from pyannote.audio import Inference

embed_model = Inference("pyannote/embedding", use_auth_token="./data/audio/voxconverse_test_wav 4/aepyx.wav")

# Step 4: For each speech segment, extract embedding
embeddings = []
for segment in speech_regions.itersegments():
    emb = embed_model.crop("./data/audio/voxconverse_test_wav 4/aepyx.wav", segment)
    embeddings.append({
        "start": segment.start,
        "end": segment.end,
        "embedding": emb.data.tolist()
    })

with open("segments.pkl", "rb") as f:
    segments = pickle.load(f)

for seg in segments:
    print(seg)