import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DummyPublisher(Node):
    def __init__(self):
        super().__init__("dummy_publisher")
        self.pub = self.create_publisher(String, "/dummy_topic", 10)
        self.timer = self.create_timer(1.0, self.tick)
        self.count = 0

    def tick(self):
        msg = String()
        msg.data = f"Hello from Docker #{self.count}"
        self.pub.publish(msg)
        self.get_logger().info(msg.data)
        self.count += 1

def main():
    rclpy.init()
    node = DummyPublisher()
    rclpy.spin(node)

if __name__ == "__main__":
    main()

