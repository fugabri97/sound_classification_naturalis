import librosa
import numpy as np

DURATION = 6
SAMPLE_RATE = 35000
N_MELLS = 128

def preprocess(file):
    signal, sr = librosa.load(file, sr=None)
    signal = librosa.util.fix_length(signal, size=int(DURATION * SAMPLE_RATE))
    mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELLS)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = mel[..., np.newaxis]
    mel = mel.transpose(1, 0, 2)
    return mel