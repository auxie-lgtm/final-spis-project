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
        # Use relative bands so the CNN receives a balanced target distribution
        # when the dataset's absolute discrepancy range is uneven.
        if not avg_discs:
            self.__rank_eval = []
            return
        groups = []
        for discrepancy in sorted(set(avg_discs)):
            groups.append((discrepancy, avg_discs.count(discrepancy)))

        grade_by_disc = {}
        grades = self.rankings
        samples_seen = 0
        for discrepancy, group_size in groups:
            grade_index = min(
                len(grades) - 1,
                samples_seen * len(grades) // len(avg_discs),
            )
            grade_by_disc[discrepancy] = grades[grade_index]
            samples_seen += group_size

        eval = [grade_by_disc[discrepancy] for discrepancy in avg_discs]
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

    


