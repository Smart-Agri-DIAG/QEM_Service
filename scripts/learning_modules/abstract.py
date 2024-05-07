import numpy as np
from abc import ABC, abstractmethod


class LearningModule(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, batch):
        pass
