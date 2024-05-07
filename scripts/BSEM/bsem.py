#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from qem_service.msg import BoundingBoxes
from cv_bridge import CvBridge
import message_filters
import math
import rospkg
import os
import sys
import torch


class BerrySizeShapeEstimationModule:
    def __init__(self):
        # rospy.wait_for_service('qem_service')
        params = rospy.get_param("/quality_service/")
        self.bridge = CvBridge()
        rospy.Subscriber(params["image_info_topic"], CameraInfo, self.colorInfoCallback)
        rospy.Subscriber(params["depth_info_topic"], CameraInfo, self.depthInfoCallback)

        rospack = rospkg.RosPack()
        cwd_path = rospack.get_path('qem_service')
        weights_path = cwd_path + "/weights/berries_detector.pt"
        params = rospy.get_param("/quality_service/")
        if os.path.exists(cwd_path + '/yolov5'):
            self.model = torch.hub.load(cwd_path + '/yolov5', 'custom', path=weights_path, source="local")
            self.model.conf = params["conf_tresh"]
        else:
            print("Get YoloV5 repository before!")

    def depthInfoCallback(self, msg):
        # Retrieve camera informations
        self.depth_info = True
        self.cX_d = msg.K[2]
        self.cY_d = msg.K[5]
        self.fx_d = msg.K[0]
        self.fy_d = msg.K[4]
        self.width_d = msg.width
        self.height_d = msg.height

    def colorInfoCallback(self, msg):
        # Retrieve camera informations
        self.color_info = True
        self.cX_rgb = msg.K[2]
        self.cY_rgb = msg.K[5]
        self.fx_rgb = msg.K[0]
        self.fy_rgb = msg.K[4]
        self.width_rgb = msg.width
        self.height_rgb = msg.height

    def berriesCallback(self, image, depth, bboxes):
        berries_size = {}
        depth = depth/1000
        # print("SSEM", bboxes.bounding_boxes)
        # print("SSEM HEADER", bboxes.header)

        for bbox in bboxes.bounding_boxes:
            # Draw bounding box
            id_ = bbox.id
            x_min = bbox.x_min
            y_min = bbox.y_min
            x_max = bbox.x_max
            y_max = bbox.y_max
            sampled_depth = depth[int((y_min-1+y_max-1)/2), int((x_min-1+x_max-1)/2)]
            patch = image[y_min:y_max, x_min:x_max]
            if not len(patch):
                continue
            pred = self.model(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), size=256)
            heights = []
            widths = []
            for bbox in pred.xyxy[0]:
                # patch = cv2.rectangle(patch, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                (height, width) = self.get_bbox_size([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], sampled_depth)
                height = height*1000
                width = width*1000
                heights.append(int(height))
                widths.append(int(width))
                font = cv2.FONT_HERSHEY_SIMPLEX
                patch = cv2.rectangle(patch, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
                patch = cv2.putText(patch, "W:" + str(int(width)), (int(bbox[0]), int(bbox[1])), font, 0.5, (0, 0, 255), 1)
            berries_size[str(id_)] = [round(np.mean(heights), 2), round(np.mean(width), 2)]

        return berries_size

    def project_pixel_to_world(self, pixel, depth):
        x = (pixel[0] - self.cX_d)*depth/self.fx_d
        y = (pixel[1] - self.cY_d)*depth/self.fy_d
        return [x, y, depth]

    def get_bbox_size(self, bbox, sampled_depth):
        """Given a bounding box and the depth image, gets its metric size (in mm)
        Args:
            bbox (list): coordinates of the bounding boxes in pixels (x_min, y_min, x_max, y_max)
            depth_image (array): depth image stored as array (opencv, numpy, ...)

        Returns:
            height, width (tuple): metric height and width (in mm) of the bounding box
            line_set (o3d.geometry.LineSet): bounding box projected in 3D to display
        """
        up_right = self.project_pixel_to_world((int(bbox[2]), int(bbox[3])), sampled_depth)
        up_left = self.project_pixel_to_world((int(bbox[0]), int(bbox[3])), sampled_depth)
        down_right = self.project_pixel_to_world((int(bbox[2]), int(bbox[1])), sampled_depth)
        down_left = self.project_pixel_to_world((int(bbox[0]), int(bbox[1])), sampled_depth)

        height = math.sqrt(math.pow(up_right[0]-down_right[0], 2) + math.pow(up_right[1]-down_right[1], 2) + math.pow(up_right[2]-down_right[2], 2))
        width = math.sqrt(math.pow(down_right[0]-down_left[0], 2) + math.pow(down_right[1]-down_left[1], 2) + math.pow(down_right[2]-down_left[2], 2))

        return (height, width)


def main():
    rospy.init_node('BerrySizeShapeEstimationModule', anonymous=True)
    BerrySizeShapeEstimationModule()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    main()
