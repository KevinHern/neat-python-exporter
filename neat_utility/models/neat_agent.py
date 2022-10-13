from abc import ABCMeta, abstractmethod


class NeatAgent(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @staticmethod
    @abstractmethod
    def log_stats(agent):
        pass
