#!/usr/bin/env python
import message_filters
import rospy
from qem_service.msg import BoundingBoxes, Qem
from qem_service.srv import qem, qemRequest, qemResponse
from qem_service.srv import qem_compressed, qem_compressedRequest, qem_compressedResponse
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import yaml
import rospkg
import sys
import json


class QualityClient():
    """
    Quality estimation client for the quality estimation server. The scope of this class
    is to test and obtain the output of the ros service, and can be used as an example of
    usage of the QEM module for agronomic decisions.
    """
    def __init__(self):
        params = rospy.get_param("/quality_service/")
        self.pub = rospy.Publisher(params["output_topic"], Qem, queue_size=10)

        if params["compressed"]:
            color_sub = message_filters.Subscriber(params["image_topic"], CompressedImage)
        else:
            color_sub = message_filters.Subscriber(params["image_topic"], Image)

        depth_sub = message_filters.Subscriber(params["depth_topic"], Image)
        bboxes_sub = message_filters.Subscriber(params["tracker_topic"], BoundingBoxes)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, bboxes_sub], 1, 0.4)
        ts.registerCallback(self.qem_callback)

    def qem_callback(self, color_msg, depth_msg, bboxes_msg):
        """
        Callback for the quality estimation server. The client is able to manage both the case of
        compressed and uncompressed images, by calling two different services depending on the case.
        Args:
            - color_msg (image message): Color image
            - depth_msg (image message): Depth image
            - bboxes_msg (bounding boxes message): Detection bounding boxes of instances detected

        Returns:
            - decision (list): The decision for which grapes are ready to be harvested, and the quality of those
                               in terms of Anomaly, Brix, Color, Size estimation
        """
        try:
            if rospy.get_param("/quality_service/compressed"):
                quality = rospy.ServiceProxy("qem_service", qem_compressed)
            else:
                quality = rospy.ServiceProxy("qem_service", qem)
            response = quality(color_msg, depth_msg, bboxes_msg)

            if response.quality == "None":
                qem_out = Qem()
                qem_out.header = color_msg.header
                qem_out.bounding_boxes = bboxes_msg.bounding_boxes
                qem_out.qem_out = response.quality
                qem_out.valid_list = []
                self.pub.publish(qem_out)
                return []
            else:
                decision = self.harvesting_decision_logic(response.quality)
                qem_out = Qem()
                qem_out.header = color_msg.header
                qem_out.bounding_boxes = bboxes_msg.bounding_boxes
                qem_out.qem_out = response.quality
                qem_out.valid_list = decision
                self.pub.publish(qem_out)
                return decision

        except rospy.ServiceException as exc:
            print("Service call failed: {}".format(exc))

    def harvesting_decision_logic(self, qem_out):
        """
        Decision logic for which grape cluster should be harvested and which not, based on Anomaly, Brix, Color and Size submodules.
        Args:
            - qem_out (str): Output of the QEM module, results of all the submodules

        Returns:
            - out (list): List that contains the ids of the grapes good enough to be harvested
        """
        out = []
        dict = json.loads(qem_out)
        ids = list(dict[list(dict.keys())[0]].keys())

        for id in ids:
            a = True
            b = True
            c = True
            s = True
            for key in dict.keys():
                if key == "Anomaly":
                    if dict[key][str(id)] == 0:
                        pass
                    else:
                        a = False
                        break
                if key == "Brix":
                    if dict[key][str(id)] > 10.0:
                        pass
                    else:
                        b = False
                        break
                if key == "Color":
                    if dict[key][str(id)] <= 3:
                        pass
                    else:
                        c = False
                        break
                elif key == "Size":
                    if dict[key][str(id)][0] > 90 and dict[key][str(id)][1] > 30:
                        pass
                    else:
                        s = False
                        break
            if a and s and b and c:
                out.append(int(id))
        return out


def main():
    print("Requesting quality informations")
    rospy.wait_for_service("qem_service")
    rospy.init_node("QEM_client", anonymous=True)
    QualityClient()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutdown")


if __name__ == "__main__":
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    main()
