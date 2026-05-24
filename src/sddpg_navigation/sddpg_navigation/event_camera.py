import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# ros_gz_bridge publishes sensor topics with BEST_EFFORT reliability.
# Subscriber QoS must match exactly — mismatched policies silently prevent connection in ROS 2.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


# This node is responsible for subscribing to raw images from the robot RGB camera,
# and producing and publishing synthetic DVS event frames.

class EventCameraNode(Node):

    def __init__(self):
        super().__init__('event_camera_node')
        self.bridge = CvBridge()
        self.threshold = 0.005  # minimum normalised brightness change (0–1) to register as an event
        self.prev_frame = None

        self.create_subscription(Image, '/camera/image_raw', self._camera_cb, _SENSOR_QOS)
        self.publisher = self.create_publisher(Image, '/camera/events', _SENSOR_QOS)

    def _camera_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        frame = frame.astype(float) / 255.0
        frame = cv2.medianBlur(frame.astype(np.float32), 3)  # median blur for noise cancellation

        if self.prev_frame is None:
            self.prev_frame = frame
            return

        difference = frame - self.prev_frame

        on_events  = (difference >  self.threshold).astype(np.uint8) * 255
        off_events = (difference < -self.threshold).astype(np.uint8)
        # Encode: ON=255, no-event=128, OFF=0
        encoded = np.full_like(on_events, 128, dtype=np.uint8)
        encoded[on_events  > 0] = 255
        encoded[off_events > 0] = 0

        self.prev_frame = frame

        event_msg = self.bridge.cv2_to_imgmsg(encoded, 'mono8')
        event_msg.header = msg.header

        self.publisher.publish(event_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EventCameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()