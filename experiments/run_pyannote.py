import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diarization.pyannote_diarization import *

AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "voxconverse_test_wav")
print(AUDIO_DIR)


def run_pyannote():
    # Load the model
    model = pyannoteDiarization()
    model.load_model()

    res_path = os.path.join(PROJECT_ROOT, "results", "pyannote")

    os.makedirs(res_path, exist_ok=True)

    for filename in os.listdir(AUDIO_DIR):
        if not filename.endswith(".wav"):
            continue

        audio_path = os.path.join(AUDIO_DIR, filename)

        diarization, file_id = model.predict(audio_path)

        rttm_path = os.path.join(res_path, file_id + ".rttm")

        model.save_diarization(diarization, rttm_path)

    model.unload_model()
