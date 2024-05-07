import numpy as np
import rospy
import sys
import rospkg
import message_filters
import json
from qem_service.msg import BoundingBoxes, BoundingBox, Qem
from sensor_msgs.msg import Image


class Grape():
    def __init__(self, id, brix, color, size) -> None:
        self.id = id
        self.brix = [brix]
        self.color = [color]
        self.size = [size]
        self.avg_brix = np.mean(self.brix)
        self.avg_color = np.mean(self.color).round()
        self.avg_size = np.mean(self.size, axis=0)

    def update_quality(self):
        self.avg_brix = np.mean(self.brix)
        self.avg_color = np.mean(self.color).round()
        self.avg_size = np.mean(self.size, axis=0)


class Map():
    def __init__(self) -> None:
        self.grapes = []
        self.seen_id = []
        params = rospy.get_param("/quality_service/")

        rospy.Subscriber(params["output_topic"], Qem, self.mapper_callback)

    def mapper_callback(self, qem_msg):
        if qem_msg.qem_out != "None":
            qem = json.loads(qem_msg.qem_out)
            self.manage_grapes(qem)
            # print("QUALITY ANSWER", qem)
        else:
            # print("No grapes have been found")
            qem = None

        # print("CURRENT GRAPES")
        # for grape in self.grapes:
        #     print(grape.id)
        #     print("BRIX", grape.brix)
        #     print("AVERAGE BRIX", grape.avg_brix)
        #     print("COLOR", grape.color)
        #     print("AVERAGE COLOR", grape.avg_color)
        #     print("SIZE", grape.size)
        #     print("AVERAGE SIZE", grape.avg_size)

    def manage_grapes(self, qem):
        keys = list(qem.keys())
        ids = list(qem[keys[0]].keys())

        for id in ids:
            if id in self.seen_id:
                self.update_grape(id, keys, qem)
            else:
                self.add_grape(id, keys, qem)
                self.seen_id.append(id)

    def update_grape(self, id, keys, qem):
        for grape in self.grapes:
            if grape.id == id:
                for key in keys:
                    if key == "Brix":
                        if len(grape.brix) > 5:
                            del grape.brix[0]
                        grape.brix.append(qem[key][str(id)])
                    if key == "Color":
                        if len(grape.color) > 5:
                            del grape.color[0]
                        grape.color.append(qem[key][str(id)])
                    if key == "Size":
                        if len(grape.size) > 5:
                            del grape.size[0]
                        grape.size.append(qem[key][str(id)])
                grape.update_quality()

    def add_grape(self, id, keys, qem):
        brix = 0
        color = 0
        size = 0
        for key in keys:
            if key == "Brix":
                brix = qem[key][str(id)]
            if key == "Color":
                color = qem[key][str(id)]
            if key == "Size":
                size = qem[key][str(id)]
        self.grapes.append(Grape(id, brix, color, size))

    def remove_grape(self, id):
        pass


def main():
    rospy.init_node("mapper node", anonymous=True)
    Map()
    rospy.spin()
    try:
        rospy.spin()
    except KeyboardInterrupt as e:
        print("Keyboard shutdown", e)


if __name__ == "__main__":
    rospack = rospkg.RosPack()
    cwd_path = rospack.get_path('qem_service')
    sys.path.append(cwd_path)
    main()
