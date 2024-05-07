#!/usr/bin/env python
import json
import sys
import os
import cv2
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from qem_service.srv import qem, qemRequest, qemResponse
from qem_service.srv import qem_compressed, qem_compressedRequest, qem_compressedResponse
from dynamic_reconfigure.server import Server
from qem_service.cfg import configuration_paramsConfig
import yaml
import torch


class QualityEstimationServer():
    """
        Service of the Quality Estimation module. It calls and loads all the submodules (anomaly, brix, color and size).
        By calling this the service is initialized, and can be called for inference by the client. It also has an exception
        for the compressed image message.
    """
    def __init__(self):
        params = rospy.get_param("/quality_service/")
        # Initialize the modules
        from scripts.learning_modules.aem import AnomalyEstimationModule
        self.aem = AnomalyEstimationModule()

        # from scripts.learning_modules.bem import BrixEstimationModule
        # self.bem = BrixEstimationModule()

        # from scripts.learning_modules.cem import ColorEstimationModule
        # self.cem = ColorEstimationModule()

        from scripts.learning_modules.bncem import BrixColorEstimationModule
        self.bcem = BrixColorEstimationModule()

        from scripts.SSEM.ssem import SizeShapeEstimationModule
        self.ssem = SizeShapeEstimationModule()

        from scripts.BSEM.bsem import BerrySizeShapeEstimationModule
        self.bsem = BerrySizeShapeEstimationModule()

        self.bridge = CvBridge()

        # Call the service
        if params["compressed"]:
            service = rospy.Service("qem_service", qem_compressed, self.qem_callback)
        else:
            service = rospy.Service("qem_service", qem, self.qem_callback)

    def qem_callback(self, req):
        """
        Main callback of the QEM module. Here all the single submodules are called to perform inference on the requested image.
        Then all the results from the single submodules are packed together into a single message.
        Args:
            - req: The request from the client
        Returns:
            - out (str): String that contains for each grape the result of each submodule
        """
        params = rospy.get_param("/quality_service/")
        if params["compressed"]:
            color_image = self.bridge.compressed_imgmsg_to_cv2(req.color_image, "passthrough")
        else:
            color_image = self.bridge.imgmsg_to_cv2(req.color_image, "passthrough")

        depth_image = self.bridge.imgmsg_to_cv2(req.depth_image, "passthrough")
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if len(req.bboxes.bounding_boxes) == 0:
            return "None"
        else:
            # batch = self.crop(color_image, req.bboxes, 256)
            batch = self.new_crop(color_image, req.bboxes, 64)
            # Prediction phase of all the modules
            out_dict = {}
            if params["anomaly"]:
                aem_dict = self.aem.predict(batch, req.bboxes)
                out_dict["Anomaly"] = aem_dict

            # if params["brix"]:
            #     bem_dict = self.bem.predict(batch, req.bboxes)
            #     out_dict["Brix"] = bem_dict
            # if params["color"]:
            #     cem_dict = self.cem.predict(batch, req.bboxes)
            #     out_dict["Color"] = cem_dict

            if params["brix"] or params["color"]:
                bem_dict, cem_dict = self.bcem.predict(batch, req.bboxes)
            if params["brix"]:
                out_dict["Brix"] = bem_dict
            if params["color"]:
                out_dict["Color"] = cem_dict

            if params["size"]:
                size_dict = self.ssem.bboxesCallback(color_image, depth_image, req.bboxes)
                out_dict["Size"] = size_dict
            if params["berries"]:
                berries_dict = self.bsem.berriesCallback(color_image, depth_image, req.bboxes)
                out_dict["Berries"] = berries_dict
            if params["position"]:
                position_dict = self.ssem.position_callback(color_image, depth_image, req.bboxes)
                out_dict["Position"] = position_dict

            if not out_dict:
                return "None"
            else:
                qem_out = json.dumps(out_dict)
                return qem_out


    def new_crop(self, image, bboxes, des_size):
        for i, bbox in enumerate(bboxes.bounding_boxes):
            patch = image[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
            height, width = patch.shape[:2]
            # Check if a box is "collapsed"
            if height == 0 or width == 0:
                continue
            new_patch = cv2.resize(patch, (des_size, des_size))
            new_patch = np.transpose(new_patch, (2, 0, 1))
            new_patch = np.expand_dims(new_patch, axis=0)
            if not 'patches' in locals():
                patches = new_patch
            else:
                patches = np.concatenate((patches, new_patch))
        return patches

    def crop(self, image, bboxes, des_size):
        """
        Function that crops the patches of image detected by the YoloV5 based detector, and rescale them to a square. Then it
        stores them in a format suitable to be converted to a PyTorch tensor.
        Args:
            - image (mat): Image on which te QEM module should perform the inference
            - bboxes (list): List containing the bboxes detected in format (x_min, y_min, x_max, y_max)
            - des_size (int): Desired dimension of the patches
        Returns:
            - patches (list): List containing the cropped patches
        """
        for i, bbox in enumerate(bboxes.bounding_boxes):
            curr_size = (bbox.x_max-bbox.x_min, bbox.y_max-bbox.y_min)
            patch = image[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
            ratio = des_size/max(curr_size)
            new_size = (int(curr_size[0]*ratio), int(curr_size[1]*ratio))
            # print("Dimensioni", bbox.y_min, bbox.y_max, bbox.x_min, bbox.x_max)
            # print("QUA", patch.shape)
            new_patch = cv2.resize(patch, new_size)
            delta = [des_size-x for x in new_size]
            top, bottom = delta[1]//2, delta[1]-(delta[1]//2)
            left, right = delta[0]//2, delta[0]-(delta[0]//2)
            new_patch = cv2.copyMakeBorder(new_patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # print(new_patch.shape, type(new_patch))
            new_patch = np.transpose(new_patch, (2, 0, 1))
            new_patch = np.expand_dims(new_patch, axis=0)
            if i == 0:
                patches = new_patch
            else:
                patches = np.concatenate((patches, new_patch))
        return patches


def callback(config, level):
    pass
    return config


def main():
    print("Launching quality estimation module")
    rospy.init_node("QEM_Node", anonymous=True)
    srv = Server(configuration_paramsConfig, callback)
    QualityEstimationServer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Module stopped")


if __name__ == "__main__":
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    os.chdir(cwd_path)
    main()
