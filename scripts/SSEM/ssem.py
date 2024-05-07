#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import numpy as np
import math
import rospkg
import sys


class SizeShapeEstimationModule:
    def __init__(self):
        # rospy.wait_for_service('qem_service')
        params = rospy.get_param("/quality_service/")
        self.bridge = CvBridge()
        rospy.Subscriber(params["image_info_topic"], CameraInfo, self.colorInfoCallback)
        rospy.Subscriber(params["depth_info_topic"], CameraInfo, self.depthInfoCallback)

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

    def bboxesCallback(self, image, depth, bboxes):
        dict_size = {}
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
            (height, width) = self.get_bbox_size([x_min, y_min, x_max, y_max], depth)
            height = height*1000
            width = width*1000
            # Doesn't takes id, but enumerate
            dict_size[str(id_)] = [round(height, 2), round(width, 2)]
        return dict_size

    def project_pixel_to_world(self, pixel, depth):
        x = (pixel[0] - self.cX_d)*depth/self.fx_d
        y = (pixel[1] - self.cY_d)*depth/self.fy_d
        return [x, y, depth]

    def get_bbox_size(self, bbox, depth_image):
        """Given a bounding box and the depth image, gets its metric size (in mm)
        Args:
            bbox (list): coordinates of the bounding boxes in pixels (x_min, y_min, x_max, y_max)
            depth_image (array): depth image stored as array (opencv, numpy, ...)

        Returns:
            height, width (tuple): metric height and width (in mm) of the bounding box
        """
        depth = depth_image[int((bbox[1]-1+bbox[3]-1)/2), int((bbox[0]-1+bbox[2]-1)/2)]

        up_right = self.project_pixel_to_world((int(bbox[2]), int(bbox[3])), depth)
        up_left = self.project_pixel_to_world((int(bbox[0]), int(bbox[3])), depth)
        down_right = self.project_pixel_to_world((int(bbox[2]), int(bbox[1])), depth)
        down_left = self.project_pixel_to_world((int(bbox[0]), int(bbox[1])), depth)

        height = math.sqrt(math.pow(up_right[0]-down_right[0], 2) + math.pow(up_right[1]-down_right[1], 2) + math.pow(up_right[2]-down_right[2], 2))
        width = math.sqrt(math.pow(down_right[0]-down_left[0], 2) + math.pow(down_right[1]-down_left[1], 2) + math.pow(down_right[2]-down_left[2], 2))
        return (height, width)

    def position_callback(self, color, depth, bboxes):
        dict_position = {}
        for bbox in bboxes.bounding_boxes:
            # Draw bounding box
            id_ = bbox.id
            x_min = bbox.x_min
            y_min = bbox.y_min
            x_max = bbox.x_max
            y_max = bbox.y_max
            position = self.get_3d_position([x_min, y_min, x_max, y_max, id_], depth)
            dict_position[str(id_)] = [float(position[0]), float(position[1]), float(position[2])]
            center_u = int((x_min-1+x_max-1)/2)
            center_v = int((y_min-1+y_max-1)/2)
            cv2.circle(color, (center_u, center_v), 2, (255, 0, 0), 3)
        # cv2.imshow("IMAGE", color)
        # cv2.waitKey(0)
        return dict_position

    def get_3d_position(self, bbox, depth_image):
        """Given bounding box and depth image, returns metric position of grapes w.r.t. camera
        Args:
            bbox (list): coordinates of the bounding boxes in pixels (x_min, y_min, x_max, y_max)
            depth_image (array): depth image stored as array (opencv, numpy, ...)
        Returns:
            position (list): metric 3D position of the bounding box center [x, y, z]
        """
        center_u = int((bbox[0]-1+bbox[2]-1)/2)
        center_v = int((bbox[1]-1+bbox[3]-1)/2)
        depth = depth_image[center_v, center_u]        
        position = self.project_pixel_to_world((center_u, center_v), depth)
        return position


def main():
    rospy.init_node('SizeShapeEstimationModule', anonymous=True)
    SizeShapeEstimationModule()
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
