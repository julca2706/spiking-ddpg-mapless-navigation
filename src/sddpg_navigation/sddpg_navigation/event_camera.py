import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Match the sensor-data QoS that ros_gz_bridge uses so rqt_image_view can subscribe
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

class EventCameraNode(Node):

    def __init__(self):
        super().__init__('event_camera_node')
        self.bridge = CvBridge()
        self.threshold = 0.003
        self.prev_frame = None

        self.create_subscription(Image, '/camera/image_raw', self._camera_cb, _SENSOR_QOS)
        self.publisher = self.create_publisher(Image, '/camera/events', _SENSOR_QOS)

    def _camera_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        frame = frame.astype(float) / 255.0
        frame = cv2.medianBlur(frame.astype(np.float32), 3)

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