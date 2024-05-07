import torch
from torch.cuda.amp import autocast
from scripts.learning_modules.abstract import LearningModule
from scripts.learning_modules.models.color.model import Net
import numpy as np
import cv2


class ColorEstimationModule(LearningModule):
    def __init__(self,
                 weights="./scripts/learning_modules/models/color/plastic_model/plastic_grapes.pt",
                 threshold=0.5,
                 run_on_cpu=False):
        if run_on_cpu or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load(weights)['model_state_dict'])
        self.model.eval()
        self.threshold = threshold

    def __preporcess(self, images):
        max_pixel_value = 255
        new_batch = []
        mean = np.array([0.43935751, 0.4074413, 0.42165834])
        std = np.array([0.22417019, 0.2418512, 0.23882174])

        for im in images:
            image = np.transpose(im, (1, 2, 0))
            image = (image - mean * max_pixel_value) / (std * max_pixel_value)
            image = np.transpose(image, (2, 0, 1))
            new_batch.append(image)

        new_batch = np.array(new_batch)
        return new_batch

    # Batch in format N,C,H,W (numpy array)
    def predict(self, batch, bboxes):
        batch = torch.Tensor(self.__preporcess(batch)).to(self.device)
        with autocast():
            cem_results = {}
            predictions = torch.sigmoid(self.model(batch))
            for pred, bbox in zip(predictions, bboxes.bounding_boxes):
                id_ = bbox.id
                if pred.item() > self.threshold:
                    cem_results[str(id_)] = 1
                else:
                    cem_results[str(id_)] = 0
        return cem_results
