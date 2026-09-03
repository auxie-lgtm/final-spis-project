# --- IMPORTS ---
# These imports provide the dataset logic, numerical arrays, audio feature extraction,
# and the TensorFlow model infrastructure used for the trained singing-grade CNN.
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import librosa
import tensorflow as tf
from class_identifier_2 import *
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# File location for the saved trained model.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "singer_grade_model.keras")


# Save the trained model to disk so it can be reused later without retraining.
def save_model(model):
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


# Remove any old saved model if you want to force a clean retrain.
def remove_saved_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"Removed stale model: {MODEL_PATH}")
    else:
        print(f"No saved model found at {MODEL_PATH}; nothing to remove.")


# Load the trained model if it already exists in the project directory.
def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


class KaraokeClassifier(ClassIdentifier):
    #
    # This class has two separate evaluation paths:
    # 1) dataset_model: a trained CNN that works on the project dataset
    # 2) standalone_estimator: a rule-based pitch/note fallback for arbitrary user files
    #
    # The dataset-trained model is NOT the same as the standalone evaluator.
    # The app should prefer the dataset model when available, then fall back to the heuristic.
    # Both outputs are approximate and should be treated as rough indicators of tonal stability,
    # not exact pitch-accuracy claims without a reference melody.

    # Use a six-class grade system so training remains stable on a smaller dataset.
    coarse_rankings = ["S", "A", "B", "C", "D", "F"]

    # Initialize the classifier and optionally remove a stale model before loading.
    def __init__(self, force_retrain=False):
        super().__init__()
        self.set_keywords(["_label.txt", ".wav"])
        self.__audio_features = []
        self.__x = np.array(None)
        self.__y = np.array(None)
        if force_retrain:
            remove_saved_model()
        self.dataset_model = load_model_if_exists()
        self.standalone_estimator = "pitch_note_heuristic"

    # Accessor methods for the internal feature collection and labels.
    def get_audio_features(self):
        return self.__audio_features

    def set_audio_features(self, audio_features):
        self.__audio_features = audio_features

    def get_x(self):
        return self.__x

    def set_x(self, audio_features, dtype = np.float32):
        self.__x = np.array(audio_features, dtype)

    def get_y(self):
        return self.__y

    def set_y(self, labels, dtype = np.float64):
        cleaned = []
        for label in labels:
            if isinstance(label, (int, np.integer)):
                cleaned.append(int(label))
            elif isinstance(label, str):
                normalized = label.strip()
                if normalized in self.rankings:
                    cleaned.append(self.rankings.index(normalized))
                else:
                    cleaned.append(np.nan)
            else:
                cleaned.append(label)
        self.__y = np.array(cleaned, dtype)
    
    def calculate_weighted_avg(self, discrepancy, point1 = 0.25, point2 = 5.0):
        perfects = []
        alrights = []
        mids = []
        for value in discrepancy:
            if value < point1:
                perfects.append(value)
            elif value < point2:
                alrights.append(value)
            else:
                mids.append(value)

        return (sum(perfects)+sum(alrights)+sum(mids))/len(discrepancy)
                    

    # Read a label file and compute an average discrepancy value.
    # These discrepancy values are the dataset's underlying signal for ranking quality.
    def find_avg_disc(self, filename):
        discrepancy = []
        with open(filename, "r") as file:
            for line in file:
                parts = line.replace(",", " ").split()
                if len(parts) <= 3:
                    continue
                try:
                    recorded_pitch = float(parts[1])
                    expected_pitch = float(parts[3])
                except ValueError:
                    continue
                if not np.isfinite(recorded_pitch) or not np.isfinite(expected_pitch):
                    continue
                discrepancy.append(abs(round(recorded_pitch - expected_pitch, 2)))
        if not discrepancy:
            return None
        return self.calculate_weighted_avg(discrepancy)

    # Convert a raw audio file into a mel spectrogram feature suitable for CNN input.
    # This is the main input representation used by the trained model.
    def load_audio_feature(self, audio_path, sample_rate=16000, n_mels=128, frame_count=256):
        profile = self.estimate_note_profile(audio_path, sample_rate=sample_rate)
        return np.array([
            min(1.0, profile["voiced_ratio"]),
            min(1.0, profile["pitch_std_semitones"] / 4.0),
            min(1.0, profile["pitch_range_semitones"] / 24.0),
            min(1.0, profile["duration_seconds"] / 30.0),
            min(1.0, profile["pitch_iqr_semitones"] / 12.0),
            min(1.0, profile["pitch_motion_semitones"] / 2.0),
            min(1.0, profile["pitch_confidence"]),
            min(1.0, profile["mean_pitch_hz"] / 1000.0),
        ], dtype=np.float32)

    # Discover all labeled dataset samples in their individual directories.
    def find_files(self):
        audio_features = []
        folders = []
        avg_discs = []
        for sample_directory in sorted(self.get_directory().iterdir()):
            if not sample_directory.is_dir():
                continue

            label_path = sample_directory / f"{sample_directory.name}{self.get_keywords()[0]}"
            audio_path = sample_directory / f"{sample_directory.name}{self.get_keywords()[1]}"
            if not audio_path.exists():
                audio_path = sample_directory / f"{sample_directory.name}.mp3"
            if not audio_path.is_file():
                continue
            average_discrepancy = self.find_avg_disc(label_path)
            if average_discrepancy is None:
                label_path.unlink()
                audio_path.unlink()
                print(f"Removed unusable sample: {audio_path.name}")
                continue
            audio_features.append(self.load_audio_feature(audio_path))
            folders.append(str(label_path))
            avg_discs.append(average_discrepancy)
        self.set_audio_features(audio_features)
        self.set_folders(folders)
        self.set_avg_discs(avg_discs)
        self.set_x(self.get_audio_features())
        self.set_y(avg_discs, dtype=np.float32)

    # Validate the dataset before training so the model is not fit on empty or malformed input.
    def check_valid_inputs(self):
        if len(self.get_x()) == 0:
            raise ValueError(f"No audio samples were found in {self.get_directory()}.")
        if len(np.unique(self.get_y())) < 2:
            raise ValueError("At least two distinct discrepancy values are required for regression.")

    # Estimate a basic note/pitch profile for a standalone audio file.
    # This provides a fallback when the file is not part of the training dataset and
    # we still want a rough evaluative signal.
    def estimate_note_profile(self, audio_path, sample_rate=16000):
        waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        duration_seconds = len(waveform) / sample_rate
        f0, voiced_flag, voiced_probability = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )

        valid = np.isfinite(f0)
        if not valid.any():
            return {
                "dominant_note": "unknown",
                "mean_pitch_hz": 0.0,
                "pitch_std_hz": 0.0,
                "pitch_std_semitones": 0.0,
                "pitch_range_hz": 0.0,
                "pitch_range_semitones": 0.0,
                "pitch_iqr_semitones": 0.0,
                "pitch_motion_semitones": 0.0,
                "pitch_confidence": 0.0,
                "voiced_ratio": 0.0,
                "duration_seconds": duration_seconds,
                "note_count": 0,
                "midi_notes": [],
            }

        pitch_values = f0[valid]
        midi_pitch_values = 69 + 12 * np.log2(pitch_values / 440.0)
        pitch_std_semitones = float(np.std(midi_pitch_values))
        robust_pitch_range = float(
            np.percentile(pitch_values, 90) - np.percentile(pitch_values, 10)
        )
        robust_semitone_range = float(
            np.percentile(midi_pitch_values, 90)
            - np.percentile(midi_pitch_values, 10)
        )
        pitch_iqr_semitones = float(
            np.percentile(midi_pitch_values, 75)
            - np.percentile(midi_pitch_values, 25)
        )
        pitch_motion_semitones = float(
            np.median(np.abs(np.diff(midi_pitch_values)))
            if len(midi_pitch_values) > 1 else 0.0
        )
        pitch_confidence = float(np.nanmean(voiced_probability))
        midi_values = np.rint(midi_pitch_values)
        midi_values = midi_values[np.isfinite(midi_values)]
        counts = Counter(int(value) for value in midi_values)
        dominant_midi = counts.most_common(1)[0][0]

        return {
            "dominant_note": librosa.midi_to_note(dominant_midi),
            "mean_pitch_hz": round(float(np.mean(pitch_values)), 2),
            "pitch_std_hz": round(float(np.std(pitch_values)), 2),
            "pitch_std_semitones": round(pitch_std_semitones, 2),
            "pitch_range_hz": round(robust_pitch_range, 2),
            "pitch_range_semitones": round(robust_semitone_range, 2),
            "pitch_iqr_semitones": round(pitch_iqr_semitones, 2),
            "pitch_motion_semitones": round(pitch_motion_semitones, 2),
            "pitch_confidence": round(pitch_confidence, 2),
            "voiced_ratio": round(float(np.mean(valid)), 2),
            "duration_seconds": duration_seconds,
            "note_count": len(counts),
            "midi_notes": [int(x) for x in midi_values],
        }

    # Collapse the detailed letter ranks down to a more readable overall set.
    # The app expects single-letter grades only, so D/F should never be emitted.
    @staticmethod
    def coarsen_grade(grade):
        if grade is None:
            return "F"
        grade = str(grade).upper()
        if grade.startswith("S"):
            return "S"
        if grade.startswith("A"):
            return "A"
        if grade.startswith("B"):
            return "B"
        if grade.startswith("C"):
            return "C"
        if grade.startswith("D"):
            return "D"
        return "F"

    @staticmethod
    def discrepancy_to_grade(discrepancy):
        if discrepancy < 0.5:
            return "S"
        if discrepancy < 1.0:
            return "A"
        if discrepancy < 2.0:
            return "B"
        if discrepancy < 4.0:
            return "C"
        if discrepancy < 7.0:
            return "D"
        return "F"

    # Heuristic fallback for arbitrary user files.
    # This does not claim exact pitch accuracy; it only offers a rough estimate based on
    # pitch variance, range, and voiced coverage.
    def estimate_standalone_grade(self, audio_path):
        profile = self.estimate_note_profile(audio_path)
        if profile["note_count"] == 0:
            # No usable pitch track means we could not confidently measure vocal stability.
            # Instead of forcing an automatic F, return a low-confidence D so noisy or spoken clips
            # are not penalized as aggressively as a truly poor singing performance.
            return "D", 0.18

        pitch_std = profile["pitch_std_semitones"]
        pitch_range = profile["pitch_range_semitones"]
        voiced_ratio = profile["voiced_ratio"]

        score = 100.0
        score -= min(20.0, pitch_std * 3.0)
        # Semitone span is register-independent and is only a modest penalty because
        # a melody can legitimately cover a wide range.
        score -= min(8.0, pitch_range * 0.4)
        score -= max(0.0, (1.0 - voiced_ratio) * 15.0)
        score -= min(20.0, max(0.0, (2.0 - profile["duration_seconds"]) * 10.0))
        score = max(0.0, min(100.0, score))

        if score >= 95:
            grade = "S"
        elif score >= 84:
            grade = "A"
        elif score >= 72:
            grade = "B"
        elif score >= 58:
            grade = "C"
        elif score >= 42:
            grade = "D"
        else:
            grade = "F"

        confidence = max(0.25, min(0.8, score / 100.0))
        return grade, confidence

    def estimate_standalone_score(self, audio_path):
        profile = self.estimate_note_profile(audio_path)
        if profile["note_count"] == 0:
            return 20.0
        score = 100.0
        score -= min(20.0, profile["pitch_std_semitones"] * 3.0)
        score -= min(8.0, profile["pitch_range_semitones"] * 0.4)
        score -= max(0.0, (1.0 - profile["voiced_ratio"]) * 15.0)
        score -= min(20.0, max(0.0, (2.0 - profile["duration_seconds"]) * 10.0))
        return max(0.0, min(100.0, score))

    # Public inference method used by the rest of the app.
    # The trained CNN is tried first if available, then the heuristic fallback is used.
    def predict_grade(self, audio_path):
        # 1. Try the trained dataset CNN first.
        try:
            audio_duration = librosa.get_duration(path=audio_path)
            feature = self.load_audio_feature(audio_path)
            feature = np.expand_dims(feature, axis=0)
        except Exception:
            return self.estimate_standalone_grade(audio_path)

        if self.dataset_model is None:
            return self.estimate_standalone_grade(audio_path)

        try:
            predicted_log_discrepancy = float(self.dataset_model.predict(feature, verbose=0)[0][0])
            if not np.isfinite(predicted_log_discrepancy):
                return self.estimate_standalone_grade(audio_path)
            predicted_discrepancy = max(0.0, float(np.expm1(predicted_log_discrepancy)))
            grade = self.discrepancy_to_grade(predicted_discrepancy)
            confidence = max(0.25, min(0.8, 1.0 / (1.0 + predicted_discrepancy)))
            return grade, confidence
        except Exception:
            # 2. If the dataset model fails, fall back to the standalone heuristic.
            return self.estimate_standalone_grade(audio_path)

    def predict_discrepancy(self, audio_path):
        if self.dataset_model is None:
            return None
        feature = np.expand_dims(self.load_audio_feature(audio_path), axis=0)
        value = float(self.dataset_model.predict(feature, verbose=0)[0][0])
        if not np.isfinite(value):
            return None
        return max(0.0, float(np.expm1(value)))


    # Train the CNN on the labeled dataset.
    # This is intentionally kept separate from the runtime evaluation flow so the app can
    # load a saved model instead of retraining on each use.
    def train(self):
        try:
            self.check_valid_inputs()
        except ValueError as e:
            print(e)
            return

        x_train, x_temp, y_train, y_temp = train_test_split(
            self.get_x(), self.get_y(), test_size=0.4, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42
        )

        print(f"\nTraining size: {len(x_train)}, Validation size: {len(x_val)}, Test size: {len(x_test)}")

        # reshape y_train, y_val, and y_test to appropriate shape
        y_train = np.array(y_train).reshape((y_train.shape[0], ))
        y_val = np.array(y_val).reshape((y_val.shape[0], ))
        y_test = np.array(y_test).reshape((y_test.shape[0], ))

        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        # Regress directly on average pitch discrepancy instead of artificial classes.
        # Log targets reduce the influence of a small number of very large errors.
        model = Sequential()
        model.add(Input(shape=(8,)))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(1, activation='linear'))
        model.summary()

        # Compile the model
        model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )

        # Train the model with early stopping to prevent overfitting.
        history = model.fit(
            x_train,
            y_train_log,
            epochs=40,
            batch_size=32,
            validation_data=(x_val, y_val_log),
            callbacks=[early_stop],
        )
        print("Training history keys:", list(history.history.keys()))
        self.dataset_model = model
        save_model(model)

        # Evaluate regression error on the holdout test set.
        predicted_test = np.maximum(0.0, np.expm1(model.predict(x_test, verbose=0).reshape(-1)))
        test_mae = float(np.mean(np.abs(predicted_test - y_test)))
        print("Test mean absolute error:", test_mae)
        return model

# Script entry point for explicit training runs only.
# This block is intentionally separate from the runtime evaluation workflow.
if __name__ == "__main__":
    k = KaraokeClassifier(True)
    k.find_files()
    k.train()