import pathlib
import abc
import augment_dataset

class ClassIdentifier:
    rankings = ["S+", "S", "S-","A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+", "F", "F-"]

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
        eval = []
        for index, discrepancy in enumerate(avg_discs):
            func = 0
            rank_index = 0
            while discrepancy >= func and rank_index < len(self.rankings) - 1:
                func += 0.25
                rank_index += 1
            eval.append(self.rankings[rank_index])
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

    


