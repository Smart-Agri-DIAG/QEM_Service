import rospy
import rospkg
import sys
import json
import os
from sensor_msgs.msg import Image, CompressedImage
from qem_service.msg import Qem
from cv_bridge import CvBridge
import message_filters
import cv2


class Visualizer():
    """
    RQT based visualizer. This class manages all the elements to be displayed.
    """
    def __init__(self):
        params = rospy.get_param("/quality_service/")
        self.bridge = CvBridge()
        self.img_pub = rospy.Publisher("/Visualizer_img", Image, queue_size=1)

        if params["compressed"]:
            color_sub = message_filters.Subscriber(params["image_topic"], CompressedImage)
        else:
            color_sub = message_filters.Subscriber(params["image_topic"], Image)

        qem_sub = message_filters.Subscriber(params["output_topic"], Qem)
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, qem_sub], 1, 0.4)
        ts.registerCallback(self.visual_callback)
        print("Loading visualizer")

    def visual_callback(self, color_msg, qem_msg):
        """
        Callback that displayys the elements of the QEM module into a single image
        Args:
            - color_msg (img msg): Image on which the QEM module inferred the results
            - qem_msg (str): Output of the QEM module, results of all the submodules
        """
        params = rospy.get_param("/quality_service")
        font = cv2.FONT_HERSHEY_SIMPLEX
        if params["compressed"]:
            color_image = self.bridge.compressed_imgmsg_to_cv2(color_msg, "passthrough")
        else:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "passthrough")

        if qem_msg.qem_out == 'None' or qem_msg.qem_out == '':
            pass
        else:
            # Get dictionary with qem output
            dict = json.loads(qem_msg.qem_out)
            # Display detection and quality based on harvesting decision logic
            for i, bbox in enumerate(qem_msg.bounding_boxes):
                # If id is in valid list, display the bounding box as green, else red
                if bbox.id in qem_msg.valid_list:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                # Draw id on each bbox
                cv2.putText(color_image, "ID:" + str(bbox.id), (int(bbox.x_min + (bbox.x_max - bbox.x_min)/2 - 20), bbox.y_max - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                # Draw rectangle on bbox
                cv2.rectangle(color_image, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color, 2)

                # Display the quality estimation results
                print(dict)
                for key in dict.keys():
                    if key == "Anomaly":
                        continue
                        # point = (int(bbox.x_min + (bbox.x_max - bbox.x_min)/2 - 40), bbox.y_min + 40)
                        # cv2.putText(color_image, "A:" + str(dict[key][str(bbox.id)]), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                    if key == "Brix":
                        continue
                        # point = (int(bbox.x_min + (bbox.x_max - bbox.x_min)/2 - 40), bbox.y_min + 40)
                        # cv2.putText(color_image, "Brix:" + str(int(dict[key][str(bbox.id)])), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                    if key == "Color":
                        continue
                        # point = (int(bbox.x_min + (bbox.x_max - bbox.x_min)/2 - 40), bbox.y_min - 10)
                        # cv2.putText(color_image, "Color:" + str(dict[key][str(bbox.id)]), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                    if key == "Size":
                        continue
                        # Height arrow
                        # cv2.arrowedLine(color_image, (int(bbox.x_min-10), int(bbox.y_max)), (int(bbox.x_min-10), int(bbox.y_min)), color, 2)
                        # cv2.arrowedLine(color_image, (int(bbox.x_min-10), int(bbox.y_min)), (int(bbox.x_min-10), int(bbox.y_max)), color, 2)
                        # # Width arrow
                        # cv2.arrowedLine(color_image, (int(bbox.x_min), int(bbox.y_max) + 10), (int(bbox.x_max), int(bbox.y_max) + 10), color, 2)
                        # cv2.arrowedLine(color_image, (int(bbox.x_max), int(bbox.y_max) + 10), (int(bbox.x_min), int(bbox.y_max) + 10), color, 2)
                        # # Height
                        # cv2.putText(color_image, str(int(dict[key][str(bbox.id)][0])), (int(bbox.x_min + 20), int(bbox.y_max + (bbox.y_min - bbox.y_max)/2)), font, 1, (0, 128, 255), 2)
                        # # Width
                        # cv2.putText(color_image, str(int(dict[key][str(bbox.id)][1])), (int(bbox.x_min + (bbox.x_max - bbox.x_min)/2 - 20), int(bbox.y_max + 40)), font, 1, (0, 128, 255), 2)

        color_image = self.bridge.cv2_to_imgmsg(color_image, "rgb8")
        self.img_pub.publish(color_image)


def main():
    rospy.init_node("Visualizer", anonymous=True)
    rospy.wait_for_service('qem_service')
    rate = rospy.Rate(0.1)
    Visualizer()
    try:
        rospy.spin()
        rate.sleep()
    except KeyboardInterrupt as e:
        print("Shutdown", e)


if __name__ == "__main__":
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    os.chdir(cwd_path)
    main()
