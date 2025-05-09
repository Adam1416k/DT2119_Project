from pyannote.audio import Pipeline
from config import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Make uniform callsign for the other methods
class pyannoteDiarization:
    """
    Class that serves as the interface for the pyannote pipeline.
    """

    def __init__(self):
        self.model_path = ""

    def load_model(self):
        """
        Load model from access token or path.
        """
        if access_token:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=access_token
            ).to(device)

        elif self.model_path != "":
            pass

    def unload_model(self):
        """
        Delete the model to free up memory.
        """
        del self.pipeline

    def predict(self, path):
        """
        Performs diarization on the given file and save the rttm.
        """

        diarization = self.pipeline(path)

        file_id = path.split("\\")[-1][:-4]  # Get file and remove .wav

        # TODO: Create the correct path for saving the predicted rttm files
        with open(file_id + ".rttm", "w") as rttm:
            diarization.write_rttm(rttm)
