'''

This is a helper file intended for audio/video processing. 
The audio has methods that 
- extract the vocal part from the audio file
- creates a plot of the frequencies displayed in the song (for user convenience)
- processes the audio and returns it to the LLM file (ai.py)

'''

# importing libraries
# os enables ffmpeg, a necessary part to managing the file
# librosa is the main audio processing file, with numpy as an assistant in managing data
# torch enables device management and tries to use something faster than a cpu to manipulate data
# matplotlib helps create the plot for user convenience
# transformers provides the model used to create the transcript
# demucs helps with extracting the vocal parts from the audio
# soundfile creates the file
import os
import numpy
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from transformers import pipeline
from demucs.apply import apply_model
from demucs.pretrained import get_model
import soundfile as sf

ffmpeg_dir = os.path.join(
    os.path.dirname(__file__),
    ".venv",
    "Lib",
    "site-packages",
    "imageio_ffmpeg",
    "binaries",
)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# TO-DO: Add a function to extract vocal from audio files. 
# Then, only the vocal function should be used in making the spectrogram. 
class AudioProcessor:
    '''
    The AudioProcessor class processes the audio into a vocal-only file and gives
    important metrics used by the LLM in the PromptManager class to evaluate the performance. 
    '''
    # The model used for audio transcription is Whisper Tiny.
    __model = "openai/whisper-tiny"

    # Creates the AudioProcessor object to use in ai.py
    def __init__(self):
        self.generate = pipeline(
            task = "automatic-speech-recognition",
            model = self.__model,
            device = 0 if torch.cuda.is_available() else -1
        )

    @staticmethod
    def pitch_statistics(f0):
        valid_f0 = f0[numpy.isfinite(f0)]
        if valid_f0.size == 0:
            return {
                "average_hz": None,
                "median_hz": None,
                "minimum_hz": None,
                "maximum_hz": None,
                "range_hz": None,
                "variation_hz": None,
                "voiced_percentage": 0.0,
            }

        minimum_hz = float(valid_f0.min())
        maximum_hz = float(valid_f0.max())
        return {
            "average_hz": round(float(valid_f0.mean()), 2),
            "median_hz": round(float(numpy.median(valid_f0)), 2),
            "minimum_hz": round(minimum_hz, 2),
            "maximum_hz": round(maximum_hz, 2),
            "range_hz": round(maximum_hz - minimum_hz, 2),
            "variation_hz": round(float(valid_f0.std()), 2),
            "voiced_percentage": round(float(valid_f0.size / f0.size * 100), 2),
        }

    # Serves to extract the voice from the audio file
    def extract_voice(self, audio_path, output_path = "extracted_voice.wav"):
        # Load pretrained model
        model = get_model("htdemucs")

        # Load audio file without requiring TorchCodec.
        wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.T)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        wav = wav.unsqueeze(0)

        # Run the separation
        with torch.no_grad():
            sources = apply_model(model, wav, split=True, shifts=1)

        vocal_index = model.sources.index("vocals")
        vocals = sources[0, vocal_index].cpu().numpy().T
        sf.write(output_path, vocals, sr)
        return output_path

    # Creates a pitch plot of the audio file and saves it as an image.
    def create_plot(self, audio_path, output_path = "plot.png"):
        waveform, sample_rate = librosa.load(audio_path, sr = 16000, mono = True)

        # Estimate the sung fundamental frequency in Hertz.
        f0, _, _ = librosa.pyin(
            waveform,
            fmin = librosa.note_to_hz("C2"),
            fmax = librosa.note_to_hz("C7"),
            sr = sample_rate,
            frame_length = 1024,
            hop_length = 256,
        )
        times = librosa.times_like(f0, sr = sample_rate, hop_length = 256)
        finite_f0 = f0[numpy.isfinite(f0)]
        display_fmax = min(
            sample_rate / 2,
            max(120, float(finite_f0.max() * 1.2)) if finite_f0.size else 120,
        )

        figure, axis = plt.subplots(figsize = (12, 4))
        axis.set_ylim(0, display_fmax)
        axis.set_xlim(0, times[-1] if times.size else 1)
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title("Sung Fundamental Frequency")

        valid = numpy.isfinite(f0)
        if valid.sum() > 1:
            points = numpy.column_stack((times[valid], f0[valid]))
            segments = numpy.stack((points[:-1], points[1:]), axis = 1)
            pitch_norm = Normalize(vmin = 40, vmax = display_fmax, clip = True)
            pitch_line = LineCollection(
                segments,
                cmap = "turbo",
                norm = pitch_norm,
                linewidth = 2.2,
                zorder = 3,
            )
            pitch_line.set_array(f0[valid][:-1])
            axis.add_collection(pitch_line)
            figure.colorbar(pitch_line, ax = axis, label = "Pitch (Hz)", pad = 0.02)

        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)

        return f0

    # Reads audio, creates its plot, and transcribes it for the LLM.
    def process_audio(self, audio_path, plot_path="plot.png"):

        voiceover_path = self.extract_voice(audio_path)
        plot = self.create_plot(voiceover_path, plot_path)
        pitch_statistics = self.pitch_statistics(plot)
        waveform, sample_rate = librosa.load(voiceover_path, sr = 16000, mono = True)

        # creates transcript
        transcript = self.generate({
            "array": waveform,
            "sampling_rate": sample_rate,
        },
            return_timestamps = True,
            chunk_length_s = 30,
            stride_length_s = (4, 2),
            generate_kwargs = {
                "condition_on_prev_tokens": False,
                "no_repeat_ngram_size": 3,
            },
        )["text"].strip()
        return {
            "transcript": transcript,
            "pitch_statistics": pitch_statistics,
            "plot_path": plot_path,
            "plot_shape": list(plot.shape),
        }

    