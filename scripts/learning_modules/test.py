import os
import cv2
from aem import AnomalyEstimationModule

def main(weights):

    check_Anomaly = AnomalyEstimationModule (models_path=weights)
    fold_path = '/home/mulham/canopies/Deep_Learning_and_PyTorch/input/train_dataset/Positive/'
    #'/media/mulham/Elements1/canopies/sickness/clasification_test/full_patch/dataset/part_o_positive/'
    #image_directory='/home/mulham/catkin_ws/src/yolov5_ros/src/yolov5/runs/detect/exp17/crops/tiled/'
    #output='/home/mulham/catkin_ws/src/yolov5_ros/src/yolov5/runs/detect/exp17/crops/heat_map/'
    images = []   
    id = [] 
    Dic = {}
    images
    #dict()
    uninfected_images = os.listdir(fold_path)
    print(uninfected_images)
    for i, image_name in enumerate(uninfected_images):
        if (image_name.split('.')[1] == 'jpeg'):
            image = cv2.imread(fold_path + image_name)
            print(i)
            images.append(image) 
            id.append(i)
        
    Dic = zip(id,images)    

    pred = check_Anomaly.predict(Dic)

    print(pred)
    # cv2.namedWindow("Starting", cv2.WINDOW_NORMAL)
    # cv2.imshow('Starting', heatmap[0])
    # cv2.waitKey(0)   
if __name__ == "__main__":
    weights= '/home/mulham/canopies/Deep_Learning_and_PyTorch/outputs/trained_models/outputs/resnet50_256_128_new/best_model.pth'
#'    '/home/mulham/canopies/Deep_Learning_and_PyTorch/outputs/trained_models/outputs/full_bunch_256_128_resnet50/best_model.pth'
    #'/home/mulham/Desktop/canopies/QEM (copy)/ml/anomaly_em/models/classification'
    main(weights)