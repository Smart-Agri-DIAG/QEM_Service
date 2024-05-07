import torch
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
import rospkg
from qem_service.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import yaml


class Detector():
    def __init__(self, model):
        """
        YoloV5 based detector for grape bunches. Is the base of the QEM module, in inference phase provides the
        bounding boxes that contain the detected grape in the form (x_min, y_min, x_max, y_max). Exception have
        been provided in case the image messages are compressed.
        """
        params = rospy.get_param("/quality_service/")
        self.model = model
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(params["bboxes_topic"], BoundingBoxes, queue_size=1)
        if params["compressed"]:
            rospy.Subscriber(params["image_topic"], CompressedImage, self.detect_callback)
        else:
            rospy.Subscriber(params["image_topic"], Image, self.detect_callback)

    def detect_callback(self, msg):
        """
        Detection callback that publishes the BoundingBoxes message every time a new image is received.
        Args:
            - msg (image message): Image message captured by a camera device
        """
        params = rospy.get_param("/quality_service/")
        self.model.conf = params["conf_tresh"]
        try:
            if params["compressed"]:
                image = self.bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            else:
                image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            pred = self.model(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), size=1280)
            bounding_boxes = BoundingBoxes()
            count_id = 0
            for bbox in pred.xyxy[0]:
                bounding_box = BoundingBox()
                bounding_box.id = count_id
                bounding_box.x_min = int(bbox[0])
                bounding_box.y_min = int(bbox[1])
                bounding_box.x_max = int(bbox[2])
                bounding_box.y_max = int(bbox[3])
                bounding_boxes.bounding_boxes.append(bounding_box)
                count_id += 1

            # Publish bounding box message
            h = Header()
            h.stamp = msg.header.stamp
            bounding_boxes.header = h
            self.publisher.publish(bounding_boxes)
            # cv2.imshow("DET IMAGE", image)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)


def main(model):
    rospy.init_node("Detector", anonymous=True)
    Detector(model)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    weights_path = cwd_path + "/weights/detector.pt"
    rospy.wait_for_service('qem_service')
    params = rospy.get_param("/quality_service/")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if os.path.exists(cwd_path + '/yolov5'):
        model = torch.hub.load(cwd_path + '/yolov5', 'custom', path=weights_path, source="local").to(device)
        model.conf = params["conf_tresh"]
    else:
        print("Get YoloV5 repository before!")
    main(model)
