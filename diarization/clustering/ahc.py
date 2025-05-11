import torch
import numpy as np
import librosa
import pickle
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio import Model, Inference
from pyannote.core import Annotation, Segment

# Load segments
with open("segments.pkl", "rb") as f:
    segments = pickle.load(f)

# Load audio
audio_path = "./data/audio/voxconverse_test_wav 4/aepyx.wav"
waveform, sample_rate = librosa.load(audio_path, sr=None)

# Load embedding model
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token="your_auth_token")
inference = Inference(embedding_model, window="whole")

# Extract embeddings
def get_embedding(segment, waveform, sample_rate):
    start_sample = int(segment["start"] * sample_rate)
    end_sample = int(segment["end"] * sample_rate)
    segment_waveform = torch.tensor(waveform[start_sample:end_sample]).unsqueeze(0)
    embedding = inference({'waveform': segment_waveform, 'sample_rate': sample_rate})
    if hasattr(embedding, "data"):
        return np.array(embedding)
    return embedding

embeddings = [get_embedding(seg, waveform, sample_rate) for seg in segments]
embeddings = np.vstack(embeddings)

# Agglomerative Hierarchical Clustering
from sklearn.metrics import silhouette_score

best_score = -1
best_n = None
for n in range(2, 10):  # Try from 2 to 9 speakers
    clustering = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, metric="cosine")
    print(f"n_clusters={n}, silhouette={score:.4f}")
    if score > best_score:
        best_score = score
        best_n = n

print(f"Best number of speakers: {best_n}")
# Create a new annotation with predicted speaker labels
annotation = Annotation()
for segment, label in zip(segments, labels):
    segment_obj = Segment(segment["start"], segment["end"])
    annotation[segment_obj] = f"Speaker {label}"

# Print annotation result
print(annotation)
