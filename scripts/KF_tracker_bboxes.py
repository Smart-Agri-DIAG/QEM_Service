import numpy as np
import cv2
import rospy
from qem_service.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from scipy.optimize import linear_sum_assignment
import message_filters
import time
import os
import rospkg
import sys


class TrackerNode():
    def __init__(self):
        self.tracks = []
        self.prev_image = []
        self.cv_bridge = CvBridge()
        self.count = 0
        self.prev_id = 0
        self.color_id = 0
        params = rospy.get_param("/quality_service/")
        self.img_pub = rospy.Publisher("/Tracking_img", Image, queue_size=1)
        self.track_publisher = rospy.Publisher(params["tracker_topic"], BoundingBoxes, queue_size=1)
        bboxes_sub = message_filters.Subscriber(params["bboxes_topic"], BoundingBoxes)
        if params["compressed"]:
            color_sub = message_filters.Subscriber(params["image_topic"], CompressedImage)
        else:
            color_sub = message_filters.Subscriber(params["image_topic"], Image)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, bboxes_sub], 1, 0.2)
        ts.registerCallback(self.tracker_callback)

    def tracker_callback(self, image_msg, bboxes_msg):
        try:
            params = rospy.get_param("/quality_service/")
            if params["compressed"]:
                img = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, "passthrough")
            else:
                img = self.cv_bridge.imgmsg_to_cv2(image_msg, "passthrough")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray_img.shape[:2]
            if self.count == 0:
                self.count = 1
                self.prev_image = gray_img.copy()
                for bbox in bboxes_msg.bounding_boxes:
                    bbox_ = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
                    # Generate one tracker for each detected bounding box
                    tracker.add_track(1, bbox_to_meas(bbox_), 0.05, 0.00625)
            else:
                Aff = tracker.camera_motion_computation(self.prev_image, gray_img)
                bboxes = []
                for bbox in bboxes_msg.bounding_boxes:
                    bboxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
                tracker.update_tracks(bboxes, Aff, img)
                self.prev_image = gray_img.copy()
            bboxes_pub = BoundingBoxes()
            for track in tracker.tracks:
                if track.display:
                    bbox = list(meas_to_bbox(track.get_state()))
                    # Check if tracker went outside borders
                    if bbox[0] > width:
                        continue
                    if bbox[2] > width:
                        bbox[2] = width
                    if bbox[1] > height:
                        bbox[1] = height
                    if bbox[3] > height:
                        continue
                    bbox_pub = BoundingBox()
                    bbox_pub.id = track.id
                    bbox_pub.x_min = max(0, bbox[0])
                    bbox_pub.y_min = max(0, bbox[1])
                    bbox_pub.x_max = max(0, bbox[2])
                    bbox_pub.y_max = max(0, bbox[3])
                    bboxes_pub.bounding_boxes.append(bbox_pub)
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track.color, 2)
                    cv2.putText(img, str(track.id), (int(bbox[0]), int(bbox[3])), cv2.FONT_HERSHEY_COMPLEX, 3, track.color, 2)
                bboxes_pub.header = bboxes_msg.header
            pub = self.cv_bridge.cv2_to_imgmsg(img, "passthrough")
            self.img_pub.publish(pub)
            self.track_publisher.publish(bboxes_pub)
        except CvBridgeError as e:
            print(e)


def main():
    # Initialize the tracker
    rospy.init_node("Tracker_node", anonymous=True)
    TrackerNode()
    try:
        rospy.spin()
    except KeyboardInterrupt as e:
        print("Keyboard shutdown", e)


if __name__ == "__main__":
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    os.chdir(cwd_path)
    rospy.wait_for_service("qem_service")
    from scripts.tracker.tracker import Tracker, bbox_to_meas, meas_to_bbox
    # Initialize tracker
    tracker = Tracker(features="optical_flow", transform="affine")
    main()
