#!/usr/bin/env python
import roslib
import unittest
import rostest
import sys
import json
import rospkg
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from qem.msg import BoundingBoxes, BoundingBox
PKG = 'qem'
NAME = 'test_qem'
roslib.load_manifest(PKG)


class TestQemNode(unittest.TestCase):
    def test_get_bbox_size_one(self):
        test_class = SizeShapeEstimationModule()
        test_depth = np.ones((20, 20))
        test_bbox = [0.0, 0.0, 0.0, 0.0]
        test_class.cX_d = 0
        test_class.cY_d = 0
        test_class.fx_d = 1
        test_class.fy_d = 1
        assert test_class.get_bbox_size(test_bbox, test_depth) == (0, 0)

    def test_get_bbox_size_two(self):
        test_class = SizeShapeEstimationModule()
        test_depth = np.ones((20, 20))
        test_bbox = [0.0, 10.0, 10.0, 10.0]
        test_class.cX_d = 0
        test_class.cY_d = 0
        test_class.fx_d = 1
        test_class.fy_d = 1
        assert test_class.get_bbox_size(test_bbox, test_depth) == (0, 10)

    def test_get_bbox_size_three(self):
        test_class = SizeShapeEstimationModule()
        test_depth = np.ones((20, 20))
        test_bbox = [10.0, 0.0, 10.0, 10.0]
        test_class.cX_d = 0
        test_class.cY_d = 0
        test_class.fx_d = 1
        test_class.fy_d = 1
        assert test_class.get_bbox_size(test_bbox, test_depth) == (10, 0)

    # def test_io(self):
    #     cv_bridge = CvBridge()
    #     test_class = QualityEstimationServer()
    #     img = cv2.imread(cwd_path + "/test/test_images/color.png")
    #     depth = cv2.imread(cwd_path + "/test/test_images/depth.png", cv2.IMREAD_GRAYSCALE)
    #     img_msg = cv_bridge.cv2_to_imgmsg(img, "passthrough")
    #     depth_msg = cv_bridge.cv2_to_imgmsg(depth, "passthrough")
    #     bboxes = BoundingBoxes()
    #     bounding_box = BoundingBox()
    #     bounding_box.id = 0
    #     bounding_box.x_min = int(0)
    #     bounding_box.y_min = int(0)
    #     bounding_box.x_max = int(20)
    #     bounding_box.y_max = int(20)
    #     bboxes.bounding_boxes.append(bounding_box)
    #     test_class.ssem.cX_d = 0
    #     test_class.ssem.cY_d = 0
    #     test_class.ssem.fx_d = 1
    #     test_class.ssem.fy_d = 1
    #     out = json.loads(test_class.qem_callback(img_msg, depth_msg, bboxes))
    #     assert list(out.keys()) == ['Anomaly', 'Brix', 'Color', 'Size']

    # def test_io2(self):
    #     cv_bridge = CvBridge()
    #     test_class = QualityEstimationServer()
    #     img = cv2.imread(cwd_path + "/test/test_images/color.png")
    #     depth = cv2.imread(cwd_path + "/test/test_images/depth.png", cv2.IMREAD_GRAYSCALE)
    #     img_msg = cv_bridge.cv2_to_imgmsg(img, "passthrough")
    #     depth_msg = cv_bridge.cv2_to_imgmsg(depth, "passthrough")
    #     bboxes = BoundingBoxes()
    #     bounding_box = BoundingBox()
    #     bounding_box.id = 0
    #     bounding_box.x_min = int(0)
    #     bounding_box.y_min = int(0)
    #     bounding_box.x_max = int(20)
    #     bounding_box.y_max = int(20)
    #     bboxes.bounding_boxes.append(bounding_box)
    #     gt_len = len(bboxes.bounding_boxes)
    #     test_class.ssem.cX_d = 0
    #     test_class.ssem.cY_d = 0
    #     test_class.ssem.fx_d = 1
    #     test_class.ssem.fy_d = 1
    #     out = json.loads(test_class.qem_callback(img_msg, depth_msg, bboxes))
    #     for value in list(out.values()):
    #         assert len(list(value.keys())) == gt_len

    # def test_crop(self):
    #     test_class = QualityEstimationServer()
    #     img = np.ones((2000, 2000, 3))
    #     bboxes = BoundingBoxes()
    #     for i in range(4):
    #         bounding_box = BoundingBox()
    #         bounding_box.id = 0
    #         bounding_box.x_min = np.random.randint(0, 1000)
    #         bounding_box.y_min = np.random.randint(0, 1000)
    #         bounding_box.x_max = bounding_box.x_min + np.random.randint(0, 1000)
    #         bounding_box.y_max = bounding_box.y_min + np.random.randint(0, 1000)
    #         bboxes.bounding_boxes.append(bounding_box)
    #     assert test_class.crop(img, bboxes, 256).shape == (4, 3, 256, 256)


if __name__ == '__main__':
    rospy.init_node("TESTS", anonymous=True)
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    sys.path.append("/home/leonardo/catkin_ws/src/QEM")
    from scripts.SSEM.ssem import SizeShapeEstimationModule
    from scripts.qem_server import QualityEstimationServer
    rostest.rosrun(PKG, NAME, TestQemNode)
