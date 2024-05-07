import torch
from torch.cuda.amp import autocast
from scripts.learning_modules.abstract import LearningModule
from scripts.learning_modules.models.brixcolor.brixcolornet import BrixColorNet
import numpy as np


class BrixColorEstimationModule(LearningModule):
    def __init__(self,
                 weights="./weights/d4.2_release.pt",
                 threshold=0.5,
                 run_on_cpu=False):
        if run_on_cpu or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        self.model = BrixColorNet().to(self.device)
        self.model.load_state_dict(torch.load(weights)['model_state_dict'])
        self.model.eval()
        self.threshold = threshold

    def __preporcess(self, images):
        return images

    # Batch in format N,C,H,W (numpy array)
    def predict(self, batch, bboxes):
        batch = torch.Tensor(self.__preporcess(batch)).to(self.device)
        # extract features:
        x, _ = self.model.auto_encoder.encode(batch)
        x = self.model.neck(x)

        color_results = dict()
        brix_results = dict()

        for pred_class, pred_regr, bbox in zip(self.model.classifier(x), self.model.regressor(x), bboxes.bounding_boxes):
            id_ = bbox.id
            color_results[str(id_)] = round(torch.argmax(pred_class).item(), 2)
            brix_results[str(id_)] = round(pred_regr.item(), 2)
        return brix_results, color_results
