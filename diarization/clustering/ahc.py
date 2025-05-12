import torch
import numpy as np
import librosa
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pyannote.audio import Model, Inference
from pyannote.core import Annotation, Segment

# Load segments
with open("segments.pkl", "rb") as f:
    segments = pickle.load(f)

# Load audio
audio_path = "./data/audio/voxconverse_test_wav 4/aepyx.wav"
waveform, sample_rate = librosa.load(audio_path, sr=None)

# Load embedding model
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token="use_auth_token")
inference = Inference(embedding_model, window="whole")

# Extract embeddings
def get_embedding(segment, waveform, sample_rate):
    start_sample = int(segment["start"] * sample_rate)
    end_sample = int(segment["end"] * sample_rate)
    segment_waveform = torch.tensor(waveform[start_sample:end_sample]).unsqueeze(0)
    embedding = inference({'waveform': segment_waveform, 'sample_rate': sample_rate})
    return np.array(embedding.data if hasattr(embedding, "data") else embedding)

embeddings = [get_embedding(seg, waveform, sample_rate) for seg in segments]
embeddings = np.vstack(embeddings)

# Agglomerative clustering
best_score = -1
best_n = None
for n in range(2, 10):  # Try clustering from 2 to 9 speakers
    clustering = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")
    print(f"n_clusters={n}, silhouette={score:.4f}")
    if score > best_score:
        best_score = score
        best_n = n

print(f"Best number of speakers: {best_n}")

# Final clustering with best_n
clustering = AgglomerativeClustering(n_clusters=best_n, metric="cosine", linkage="average")
labels = clustering.fit_predict(embeddings)

# Build annotation
annotation = Annotation()
for seg, label in zip(segments, labels):
    segment_obj = Segment(seg["start"], seg["end"])
    annotation[segment_obj] = f"Speaker {label}"

print(annotation)
