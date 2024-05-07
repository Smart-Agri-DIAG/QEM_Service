import numpy as np
import random
from scripts.learning_modules.abstract import LearningModule
from scripts.learning_modules.models.brix.grapefeatureextractor import GrapeFeatureExtractor
from joblib import load
from pathlib import Path


class BrixEstimationModule(LearningModule):
    def __init__(self):
        self.extractor = GrapeFeatureExtractor()
        self.cross = [8, 24, 8, 24]
        self.n_bin_x = 32
        self.n_bin_y = 32
        self.model = load("./scripts/learning_modules/models/brix/lr_v1.joblib")

    def __preporcess(self, images):
        image_list = [np.asarray(im).transpose((1, 2, 0)) for im in images]
        X = self.extractor.extract(image_list, self.n_bin_y, self.n_bin_x, cross=self.cross, hsv=True)
        return X

    def predict(self, batch, bboxes):
        X = self.__preporcess(batch)
        y_hat = self.model.predict(X)
        out = dict()
        for pred, bbox in zip(y_hat, bboxes.bounding_boxes):
            id_ = bbox.id
            out[str(id_)] = pred
        return out
