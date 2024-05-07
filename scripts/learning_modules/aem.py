import scipy
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from scripts.learning_modules.models.anomaly.model import build_model
from scripts.learning_modules.abstract import LearningModule
CLASS_NAMES = ['Healthy', 'Anomaly']

class AnomalyEstimationModule(LearningModule):
    def __init__(self, models_path="./weights", only_CPU=False ,img_h=256, img_w=128):
        self.img_h = 256
        self.img_w = 128

        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not only_CPU else 'cpu')
        self.checkpoint = torch.load(models_path + '/anomaly.pth', map_location=torch.device(self.device))

        self.model =  build_model(
                            pretrained=False,
                            fine_tune=False).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.thresh = 0.75
        
    def get_heatmap(self, pred_class, pimg, img):
        last_layer_weights = self.model.layers[-1].get_weights()[0]
        last_layer_weights_pred = last_layer_weights[:, pred_class]
        block5_conv3 = self.model.get_layer("block5_conv3").output
        last_conv_model = Model(self.model.input, block5_conv3)
        last_conv_output = last_conv_model.predict(
                                pimg[np.newaxis, :, :, :])

        last_conv_output = np.squeeze(last_conv_output)
        h = int(pimg.shape[0]/last_conv_output.shape[0])
        w = int(pimg.shape[1]/last_conv_output.shape[1])
        up_last_conv = scipy.ndimage.zoom(
            last_conv_output, (h, w, 1), order=1)

        newshape = pimg.shape[0]*pimg.shape[1]
        up_last_conv = up_last_conv.reshape((newshape, 512))
        activation = np.dot(up_last_conv,
                            last_layer_weights_pred)

        activation = activation.reshape(pimg.shape[0], pimg.shape[1])
        chp = activation.copy()
        numer = chp - np.min(chp)
        denom = (chp.max() - chp.min()) + 1e-8
        chp = numer / denom
        activation = (chp * 255).astype("uint8")
        colormap = cv2.COLORMAP_JET
        activation = cv2.applyColorMap(activation, colormap)
        activation = cv2.resize(activation, (img.shape[1], img.shape[0]))
        output = cv2.addWeighted(img, 0.7, activation, 0.3, 0)
        return output


    def preprocess(self,images):
        batch = []
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])
        for img in images:
            img = np.transpose(img, (1, 2, 0))
            image = test_transform(img)
            batch.append(image)
        batch = torch.stack(batch)

        return batch


    def predict(self, batch, bboxes):
        batch = torch.Tensor(self.preprocess(batch)).to(self.device)
        anomaly_dic = {}
        self.model.eval()
        predictions = torch.sigmoid(self.model(batch))
        for pred, bbox in zip(predictions, bboxes.bounding_boxes):
            print("AHHHHH\n", pred)
            id_ = bbox.id
            predicted_classes = 1 if pred[1] > self.thresh else 0
            anomaly_dic[str(id_)] = predicted_classes
        return anomaly_dic
