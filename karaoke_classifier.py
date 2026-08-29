#---IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import librosa
import tensorflow as tf
from class_identifier_2 import *
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class KaraokeClassifier(ClassIdentifier):
    #---CREATE CLASSES

    def __init__(self):
        super().__init__()
        self.set_keywords(["_label.txt", ".mp3"])
        self.__audio_features = []
        self.__x = np.array(None)
        self.__y = np.array(None)

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

    def check_valid_inputs(self):
        if len(self.get_x()) == 0:
            raise ValueError(f"No audio samples were found in {self.get_directory()}.")
        if len(np.unique(self.get_y())) < 2:
            raise ValueError("At least two performance classes are required for a stratified split.")


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

        # Define the CNN model -- Trial and error qty. layers to fine-tune
        model = Sequential()
        model.add(Input(shape=(128, 256, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu')) # notice no MaxPool after this one
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(class_count, activation='softmax'))
        model.summary()

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train_oh, epochs=10, batch_size=32, validation_data=(x_val, y_val_oh))

        # Evaluate the model
        y_pred = model.predict(x_test)
        y_pred = tf.argmax(y_pred, axis=1)

        loss, acc = model.evaluate(x_test, y_test_oh)
        print("Test accuracy:", acc)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

k = KaraokeClassifier()
k.find_files()
k.train()
        