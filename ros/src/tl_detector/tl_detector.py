#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import time
import numpy as np

STATE_COUNT_THRESHOLD = 2
TL_DETECTION_DISTANCE = 120 # number of waypoints before the next traffic light where traffic light is enabled
SAVE_TRAFFIC_LIGHT_IMG = False # Save traffic images to train classifier model.

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.has_image = False
        self.lights = []

        self.bridge = CvBridge()
        self.light_classifier = None
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_2d = None
        self.waypoint_tree = None

        self.img_count = 0
        
        # Set if using real car
        self.is_site = False

        self.light_classifier = TLClassifier(self.is_site)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # May want to use image_raw instead for classifier?
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        detector_rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.find_traffic_lights()
            detector_rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # Setup the Kd Tree which has log(n) complexity
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def save_img(self, light, state):
        # Thanks to https://github.com/ericlavigne/CarND-Capstone-Wolf-Pack for the idea to save images in this way
        # To build up the training set

        if SAVE_TRAFFIC_LIGHT_IMG and self.has_image: # self.img_count to reduce image save

            file_name = "IMG_" + str(time.time()).replace('.','') + '.jpg'
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            img_path = '/home/student/Udacity/tl_classifier/training_data/light_classification/IMGS/'

            if state == 0:
                if self.img_count % 10 == 0:
                    img_path = img_path+"RED/" + file_name
                    cv2.imwrite(img_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    rospy.loginfo("Path: {0}".format(img_path))
            elif state == 1:
                img_path = img_path+"YELLOW/" + file_name
                cv2.imwrite(img_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                rospy.loginfo("Path: {0}".format(img_path))
            elif state == 2:
                if self.img_count % 10 == 0:
                    img_path = img_path+"GREEN/" + file_name
                    cv2.imwrite(img_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    rospy.loginfo("Path: {0}".format(img_path))
            else:
                img_path = img_path+"UNKNOWN/" + file_name
                cv2.imwrite(img_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                rospy.loginfo("Path: {0}".format(img_path))

            self.img_count += 1

    def image_cb(self, msg):
        """ updates the current image
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        # Process traffic lights?
        
    def find_traffic_lights(self):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """

        #if self.light_classifier and self.has_image == True:
        #    if self.is_site:# site package
                # save real track images.
        #        self.save_img(self.camera_image, 4)

        light_wp, state = self.process_traffic_lights()
            # rospy.logwarn("Closest light wp: {0} \n And light state: {1}".format(light_wp, state))


        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        closest_idx = None
        if self.waypoint_tree:
            closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Assume RED
        TLstate = 0 

        if (self.has_image):
            try:
                # Convert image into something usable
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB);
                image = cv2.resize(image, (32,64))
                image_array = np.asarray(image)

                # Get classification
                TLstate = self.light_classifier.get_classification(image_array[None, :, :, :])
                #rospy.loginfo("TL State: {0}, Actual State: {1}".format(TLstate, light.state))
            except:
                #rospy.loginfo("Could not identify TL State - assuming RED")
                TLstate = 0;
            

        # Return light state
        return TLstate
        # for testing we return the light state from the simulator
        #return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        # stop_line_positions = self.config['stop_line_positions']
        # if(self.pose):
        #    car_position = self.get_closest_waypoint(self.pose.pose)

        # TODO find the closest visible traffic light (if one exists)

        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.waypoints:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # if can not get the car pose , break
            if not car_wp_idx:
               return -1, TrafficLight.UNKNOWN


            min_diff = min(len(self.waypoints.waypoints), TL_DETECTION_DISTANCE)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1]) # x,y
                # Find closest step line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < min_diff:
                    min_diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
                
                    # DEBUG
                    if d < 70: # when close traffic light , save image
                        # To save training image
                        self.save_img(self.camera_image, light.state)

        if closest_light:
            state = self.get_light_state(closest_light)
            #rospy.loginfo('closest light wp %i with color %i', line_wp_idx, state )
            return line_wp_idx, state
        else:
            return -1, TrafficLight.UNKNOWN
                  

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')