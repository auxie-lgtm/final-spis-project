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
MODEL_PATH = "singer_grade_model.keras"


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
        self.set_keywords(["_label.txt", ".mp3"])
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
        # Grabbing the first and third columns (indexes 0 and 2)
        column_1 = []
        column_3 = []
        discrepancy = []

        with open(filename, "r") as file:
            for line in file:
                # split() automatically handles spaces and tabs, while strip() removes whitespace at the ends
                parts = line.strip().split()
                
                # Ensure the line isn't empty and has enough columns
                if len(parts) > 3:
                    column_1.append(parts[1])
                    column_3.append(parts[3])
                
        for i in range(len(column_1)):
            discrepancy.append(abs(round((float(column_1[i]) - float(column_3[i])), 2)))

        return self.calculate_weighted_avg(discrepancy)

    # Convert a raw audio file into a mel spectrogram feature suitable for CNN input.
    # This is the main input representation used by the trained model.
    def load_audio_feature(self, audio_path, sample_rate=16000, n_mels=128, frame_count=256):
        waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        mel_feature = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        mel_feature = librosa.power_to_db(mel_feature, ref=np.max)
        mel_feature = librosa.util.fix_length(mel_feature, size=frame_count, axis=1)
        mel_feature = np.clip((mel_feature + 80.0) / 80.0, 0.0, 1.0)
        return mel_feature[..., np.newaxis].astype(np.float32)

    # Discover all labeled dataset samples and transform them into training data.
    # Each sample directory is expected to contain:
    # - a label file ending in _label.txt
    # - an audio file ending in .mp3
    def find_files(self):
        audio_features = []
        folders = []
        avg_discs = []
        for sample_directory in sorted(self.get_directory().iterdir()):
            if not sample_directory.is_dir():
                continue
            label_path = sample_directory / (f"{sample_directory.name}" + self.get_keywords()[0])
            audio_path = sample_directory / (f"{sample_directory.name}" + self.get_keywords()[1])
            if not audio_path.exists() or not label_path.exists():
                continue
            average_discrepancy = self.find_avg_disc(label_path)
            audio_features.append(self.load_audio_feature(audio_path))
            folders.append(str(label_path))
            avg_discs.append(average_discrepancy)
        self.set_audio_features(audio_features)
        self.set_folders(folders)
        self.set_avg_discs(avg_discs)
        self.set_rank_eval(avg_discs)
        self.set_x(self.get_audio_features())
        self.set_y(self.get_rank_eval())

    # Validate the dataset before training so the model is not fit on empty or malformed input.
    def check_valid_inputs(self):
        if len(self.get_x()) == 0:
            raise ValueError(f"No audio samples were found in {self.get_directory()}.")
        if len(np.unique(self.get_y())) < 2:
            raise ValueError("At least two performance classes are required for a stratified split.")

    # Estimate a basic note/pitch profile for a standalone audio file.
    # This provides a fallback when the file is not part of the training dataset and
    # we still want a rough evaluative signal.
    def estimate_note_profile(self, audio_path, sample_rate=16000):
        waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
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
                "pitch_range_hz": 0.0,
                "voiced_ratio": 0.0,
                "note_count": 0,
                "midi_notes": [],
            }

        pitch_values = f0[valid]
        midi_values = np.rint(69 + 12 * np.log2(pitch_values / 440.0))
        midi_values = midi_values[np.isfinite(midi_values)]
        counts = Counter(int(value) for value in midi_values)
        dominant_midi = counts.most_common(1)[0][0]

        return {
            "dominant_note": librosa.midi_to_note(dominant_midi),
            "mean_pitch_hz": round(float(np.mean(pitch_values)), 2),
            "pitch_std_hz": round(float(np.std(pitch_values)), 2),
            "pitch_range_hz": round(float(np.max(pitch_values) - np.min(pitch_values)), 2),
            "voiced_ratio": round(float(np.mean(voiced_flag[valid])) if voiced_flag.size else 0.0, 2),
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

        pitch_std = profile["pitch_std_hz"]
        pitch_range = profile["pitch_range_hz"]
        voiced_ratio = profile["voiced_ratio"]
        note_count = profile["note_count"]

        score = 100.0
        score -= min(30.0, pitch_std / 6.0)
        score -= min(25.0, pitch_range / 35.0)
        score -= max(0.0, (1.0 - voiced_ratio) * 25.0)
        score += min(15.0, note_count * 1.5)
        score = max(0.0, min(100.0, score))

        if score >= 90:
            grade = "S"
        elif score >= 80:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 30:
            grade = "D"
        else:
            grade = "F"

        confidence = max(0.25, min(0.8, score / 100.0))
        return grade, confidence

    # Public inference method used by the rest of the app.
    # The trained CNN is tried first if available, then the heuristic fallback is used.
    def predict_grade(self, audio_path):
        # 1. Try the trained dataset CNN first.
        try:
            feature = self.load_audio_feature(audio_path)
            feature = np.expand_dims(feature, axis=0)
        except Exception:
            return self.estimate_standalone_grade(audio_path)

        if self.dataset_model is None:
            return self.estimate_standalone_grade(audio_path)

        try:
            probabilities = self.dataset_model.predict(feature, verbose=0)[0]
            class_index = int(np.argmax(probabilities))
            raw_grade = self.rankings[class_index]
            grade = self.coarsen_grade(raw_grade)
            confidence = float(probabilities[class_index])
            return grade, confidence
        except Exception:
            # 2. If the dataset model fails, fall back to the standalone heuristic.
            return self.estimate_standalone_grade(audio_path)


    # Train the CNN on the labeled dataset.
    # This is intentionally kept separate from the runtime evaluation flow so the app can
    # load a saved model instead of retraining on each use.
    def train(self):
        try:
            self.check_valid_inputs()
        except ValueError as e:
            print(e)
            return

        y_numeric = np.asarray(self.get_y(), dtype=np.int64)
        class_counts = np.bincount(y_numeric)

        if np.min(class_counts[class_counts > 0]) < 2:
            print("Warning: some classes have fewer than 2 samples; using a non-stratified split.")
            x_train, x_temp, y_train, y_temp = train_test_split(self.get_x(), self.get_y(), test_size=0.4, random_state=42)
            x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
        else:
            # split data into training, validation, and test sets
            x_train, x_temp, y_train, y_temp = train_test_split(self.get_x(), self.get_y(), test_size=0.4, random_state=42, stratify=self.get_y())
            x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print(f"\nTraining size: {len(x_train)}, Validation size: {len(x_val)}, Test size: {len(x_test)}")

        # reshape y_train, y_val, and y_test to appropriate shape
        y_train = np.array(y_train).reshape((y_train.shape[0], ))
        y_val = np.array(y_val).reshape((y_val.shape[0], ))
        y_test = np.array(y_test).reshape((y_test.shape[0], ))

        class_count = len(self.rankings)

        # convert labels to one-hot encoding
        y_train_oh = tf.keras.utils.to_categorical(y_train, class_count)
        y_val_oh = tf.keras.utils.to_categorical(y_val, class_count)
        y_test_oh = tf.keras.utils.to_categorical(y_test, class_count)

        # Define a simpler CNN to reduce memorization on limited data.
        model = Sequential()
        model.add(Input(shape=(128, 256, 1)))
        model.add(Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(0.4))
        model.add(Dense(class_count, activation='softmax'))
        model.summary()

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
        )

        # Train the model with early stopping to prevent overfitting.
        history = model.fit(
            x_train,
            y_train_oh,
            epochs=20,
            batch_size=16,
            validation_data=(x_val, y_val_oh),
            callbacks=[early_stop],
        )
        print("Training history keys:", list(history.history.keys()))
        self.dataset_model = model
        save_model(model)

        # Evaluate the trained model on the holdout test set.
        y_pred = model.predict(x_test)
        y_pred = tf.argmax(y_pred, axis=1)

        loss, acc = model.evaluate(x_test, y_test_oh)
        print("Test accuracy:", acc)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        return model

# Script entry point for explicit training runs only.
# This block is intentionally separate from the runtime evaluation workflow.
if __name__ == "__main__":
    k = KaraokeClassifier(True)
    k.find_files()
    k.train()