from pyannote.audio import Pipeline
from pyannote.core import Timeline


class PyannoteSegmenter:
    """
    Segmentation module using a pre-trained Pyannote model.

    This class wraps the Hugging Face pipeline 'pyannote/segmentation' to split
    an input audio file into homogeneous speaker segments.

    Attributes:
        pipeline (pyannote.audio.Pipeline): Loaded segmentation pipeline.
    """

    def __init__(self, model_name: str = "pyannote/segmentation", device: str = "cpu"):
        """
        Initialize the segmenter with a pre-trained model.

        Args:
            model_name (str): Hugging Face model identifier for segmentation.
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        """
        # Load the pre-trained segmentation pipeline
        self.pipeline = Pipeline.from_pretrained(model_name, device=device)

    def segment(self, audio_path: str) -> Timeline:
        """
        Perform segmentation on an audio file.

        Args:
            audio_path (str): Path to the audio file to segment.

        Returns:
            Timeline: A pyannote.core.Timeline object containing speech segments.

        Example:
            segmenter = PyannoteSegmenter(device='cuda')
            timeline = segmenter.segment('path/to/audio.wav')
            for segment in timeline:
                print(segment.start, segment.end)
        """
        # Run segmentation; the pipeline returns a Timeline of segments
        segmentation: Timeline = self.pipeline({"audio": audio_path})
        return segmentation
