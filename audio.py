'''

This is a helper file intended for audio/video processing. 

'''

import os
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import pipeline

ffmpeg_dir = os.path.join(
    os.path.dirname(__file__),
    ".venv",
    "Lib",
    "site-packages",
    "imageio_ffmpeg",
    "binaries",
)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

class AudioProcessor:
    # The model used for audio transcription is Whisper Tiny.
    model = "openai/whisper-tiny.en"

    # Creates the AudioProcessor object to use in ai.py
    def __init__(self):
        self.generate = pipeline(
            task = "automatic-speech-recognition",
            model = self.model,
            device = 0 if torch.cuda.is_available() else -1
        )

    # Creates a spectrogram of the audio file and saves it as an image. 
    # This allows the LLM to read the file and analyze the audio performance. 
    def create_spectrogram(self, audio_path, output_path="spectrogram.png"):
        # loads the audio file and creates the mel spectrogram, 
        # which is then converted to decibels and saved as an image.
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
        )
        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=float(mel_spectrogram.max()))

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            decibel_spectrogram,
            sr=sample_rate,
            hop_length=256,
            x_axis="time",
            y_axis="mel",
        )

        # Adds a color bar to the spectrogram and saves the image.
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return decibel_spectrogram

    # Reads audio, creates its spectrogram, and transcribes it for the LLM.
    def process_audio(self, audio_path, spectrogram_path="spectrogram.png"):

        # creates spectrogram
        spectrogram = self.create_spectrogram(audio_path, spectrogram_path)
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

        # creates transcript
        transcript = self.generate({
            "array": waveform,
            "sampling_rate": sample_rate,
        }, return_timestamps=True, generation_config=None)["text"].strip()
        # generation_config parameter should be edited, since it's deprecated
        return {
            "transcript": transcript,
            "spectrogram_path": spectrogram_path,
            "spectrogram_shape": list(spectrogram.shape),
        }

    