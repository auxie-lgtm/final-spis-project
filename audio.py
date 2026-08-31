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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from demucs.apply import apply_model
from demucs.pretrained import get_model
import soundfile as sf

# handles file path management
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
    '''
    AudioProcessor handles the raw audio pipeline for the karaoke evaluator.

    It is responsible for converting uploaded audio into a clean signal representation,
    extracting useful vocal features, and turning those features into summary metrics
    that the LLM can explain to the user.
    '''
    # Whisper Tiny is used for speech-to-text transcription of the vocal track.
    __model = "openai/whisper-tiny"

    # Constructor: load the Whisper processor and model once, then reuse them for all files.
    # This keeps the runtime simple and avoids re-downloading or reinitializing the model each time.
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.__model)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.__model).to(self.device)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")

    # Helper for summarizing pitch information extracted from the waveform.
    # This produces a compact dictionary containing statistics such as mean pitch,
    # pitch range, and voiced percentage that can be shown to the user or fed to the LLM.
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

    # These scoring helpers convert raw audio measurements into rough 0-100 values.
    # The goal is not to be a scientifically exact rating system, but to provide an
    # interpretable signal for the LLM and the user.
    @staticmethod
    def _score_pitch(pitch_std, pitch_range, voiced_ratio):
        score = 100.0
        score -= min(35.0, pitch_std / 12.0)
        score -= min(25.0, pitch_range / 40.0)
        score -= max(0.0, (1.0 - voiced_ratio) * 30.0)
        return max(0.0, min(100.0, score))

    @staticmethod
    def _score_beat(tempo_bpm):
        if tempo_bpm <= 0:
            return 50.0
        if 70 <= tempo_bpm <= 160:
            return 85.0
        if 50 <= tempo_bpm <= 200:
            return 75.0
        return 60.0

    @staticmethod
    def _score_clarity(centroid_mean, flatness_mean):
        score = 55.0
        score += min(20.0, centroid_mean / 120.0)
        score += min(25.0, (1.0 - flatness_mean) * 60.0)
        return max(0.0, min(100.0, score))

    @staticmethod
    def _score_consistency(pitch_std, energy):
        score = 100.0
        score -= min(40.0, pitch_std / 10.0)
        score -= max(0.0, (1.0 - min(1.0, energy * 2.0)) * 20.0)
        return max(0.0, min(100.0, score))

    # Evaluate the audio file by computing signal-based indicators of vocal quality.
    # This method is one of the core parts of the product: it produces objective metrics
    # even when no dataset label or reference melody is available.
    def evaluate_audio_metrics(self, audio_path):
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )

        valid = numpy.isfinite(f0)
        if valid.any():
            pitch_values = f0[valid]
            pitch_std = float(numpy.std(pitch_values))
            mean_pitch = float(numpy.mean(pitch_values))
            pitch_range = float(numpy.max(pitch_values) - numpy.min(pitch_values))
            voiced_ratio = float(numpy.mean(voiced_flag[valid])) if voiced_flag.size else 0.0
        else:
            pitch_std = 0.0
            mean_pitch = 0.0
            pitch_range = 0.0
            voiced_ratio = 0.0

        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        tempo_values, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate, onset_envelope=onset_env)
        tempo_array = numpy.asarray(tempo_values).ravel()
        tempo_bpm = float(tempo_array[0]) if tempo_array.size > 0 else 0.0

        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
        flatness = librosa.feature.spectral_flatness(y=waveform)
        centroid_mean = float(numpy.mean(centroid))
        flatness_mean = float(numpy.mean(flatness))

        energy = float(numpy.sqrt(numpy.mean(waveform ** 2)))

        return {
            "pitch_score": round(self._score_pitch(pitch_std, pitch_range, voiced_ratio), 1),
            "beat_score": round(self._score_beat(tempo_bpm), 1),
            "clarity_score": round(self._score_clarity(centroid_mean, flatness_mean), 1),
            "consistency_score": round(self._score_consistency(pitch_std, energy), 1),
            "mean_pitch_hz": round(mean_pitch, 1),
            "pitch_range_hz": round(pitch_range, 1),
            "tempo_bpm": round(tempo_bpm, 1),
        }

    # Separate the vocal track from the original audio file.
    # Demucs is a pretrained source-separation model that isolates the vocal stem from a mixed song.
    # This is useful because we want the transcription and pitch analysis to focus on the singer's voice,
    # rather than the accompaniment or background music.
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

    # Generate a visual pitch contour plot for the user.
    # This is mainly a debugging and UX aid: it shows how the sung pitch changes over time.
    # The plot is saved to disk and can be displayed in the app if needed.
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

    # Full audio-processing pipeline used by the evaluation system.
    # It performs the end-to-end workflow:
    # 1. isolate vocals
    # 2. generate a pitch plot
    # 3. extract pitch statistics
    # 4. transcribe the vocal track with Whisper
    # 5. return a structured dictionary for the LLM to interpret.
    def process_audio(self, audio_path, plot_path="plot.png"):

        voiceover_path = self.extract_voice(audio_path)
        plot = self.create_plot(voiceover_path, plot_path)
        pitch_statistics = self.pitch_statistics(plot)
        waveform, sample_rate = librosa.load(voiceover_path, sr = 16000, mono = True)

        encoded = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = {k: v.to(self.device) for k, v in encoded.items()}

        generated_ids = self.model.generate(
            **input_features,
            forced_decoder_ids=self.processor.get_decoder_prompt_ids(language="en", task="transcribe"),
            max_new_tokens=128,
            do_sample=False,
            no_repeat_ngram_size=3,
            return_timestamps=False,
        )
        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return {
            "transcript": transcript,
            "pitch_statistics": pitch_statistics,
            "plot_path": plot_path,
            "plot_shape": list(plot.shape),
        }

    