import os, sys
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_ahc import *
from experiments.run_bhmm import *
from experiments.run_spectral import *
from experiments.run_pyannote import *
from diarization.segmenter import *

all_data = process_audio()

run_ahc(all_data)
run_bhmm(all_data)
run_spectral(all_data)

del all_data

run_pyannote()
