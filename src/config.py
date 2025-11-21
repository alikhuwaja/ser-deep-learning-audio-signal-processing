from pathlib import Path

#DATA folders path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR= PROJECT_ROOT / "data"
RAVDESS_CSV= DATA_DIR / "ravdess_index.csv"
CREMAD_CSV= DATA_DIR / "cremad_index.csv"
RAVDESS_VIDEO_CSV =  DATA_DIR / "ravdess_video_index.csv"

SAMPLE_RATE= 16000
NUM_MEL= 64
#Number of Mel frequency bands in the spectogram
NUM_FFT= 1024
# window size for the FFT when computing the spectogram
HOP_LENGTH= 512
#Step size  between the consecutive windows
SEGMENT_SEC=3

NUM_CLASSES = 6         
BATCH_SIZE = 32
N_EPOCHS = 20
LEARNING_RATE = 1e-3

# Training and valuaton data spliting 
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2