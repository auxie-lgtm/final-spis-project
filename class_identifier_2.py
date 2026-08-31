import pathlib
import abc
import augment_dataset

class ClassIdentifier:
    # Keep the dataset labels coarse to avoid overfitting on a tiny dataset.
    # Six coarse classes are much more stable than 18 fine-grained letter rankings.
    rankings = ["S", "A", "B", "C", "D", "F"]

    def __init__(self, dataset_dir=None):
        base_dir = pathlib.Path(__file__).resolve().parent / "sight-singing-vocal-data"
        if dataset_dir is not None:
            self.__directory = pathlib.Path(dataset_dir)
        else:
            self.__directory = augment_dataset.get_augmented_dataset()
        self.__keywords = ["label"]
        self.__folders = []
        self.__avg_discs = []
        self.__rank_eval = []

    def get_directory(self):
        return self.__directory

    def get_folders(self):
        return self.__folders

    def set_folders(self, folders):
        self.__folders = folders

    def get_keywords(self):
        return self.__keywords

    def set_keywords(self, keywords):
        self.__keywords = keywords

    def get_avg_discs(self):
        return self.__avg_discs

    def set_avg_discs(self, avg_discs):
        self.__avg_discs = avg_discs

    def get_rank_eval(self):
        return self.__rank_eval

    def set_rank_eval(self, avg_discs):
        # Use a wider, more even scale to spread labels across all six bands.
        # The goal is not to be mathematically perfect; it is to ensure the model sees
        # all classes and does not collapse into just a few dominant bins.
        thresholds = [0.5, 1.0, 2.0, 4.0, 7.0]
        eval = []
        for discrepancy in avg_discs:
            if discrepancy < thresholds[0]:
                grade = "S"
            elif discrepancy < thresholds[1]:
                grade = "A"
            elif discrepancy < thresholds[2]:
                grade = "B"
            elif discrepancy < thresholds[3]:
                grade = "C"
            elif discrepancy < thresholds[4]:
                grade = "D"
            else:
                grade = "F"
            eval.append(grade)
        self.__rank_eval = eval

    @abc.abstractmethod
    def calculate_weighted_avg(self, discrepancy, point1, point2):
        pass

    @abc.abstractmethod
    def find_avg_disc(self, filename):
        pass

    @abc.abstractmethod
    def find_files(self):
        pass

    


