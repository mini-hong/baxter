import rospy

import argparse
import struct
import sys
import copy
import rospy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

import numpy as np

import os
from baxter_core_msgs.msg import EndpointState
import baxter_interface
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
robot_pose, robot_orientation = [], []
img = None

class baxter(object):
	def __init__(self, hover_distance=0.1):

		rospy.init_node("obin_click2pickplace")
		rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, pos_callback)
		rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback, queue_size=1)

		self.act_dim = 5
		self.obs_dim = 12
		self.ns = "ExternalTools/right/PositionKinematicsNode/IKService"
		self._iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK)
		self._ikreq = SolvePositionIKRequest()
		self._limb = baxter_interface.Limb('right')
		self._gripper = baxter_interface.Gripper('right')
		self.hover_distance = hover_distance
		self.safety_distance = 0.01
		self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
		self._init_state = self._rs.state().enabled
		self._verbose = False
		print("Enabling robot... ")
		self._rs.enable()
		self.wait = 3.0
		rospy.wait_for_service(self.ns, 5.0)

	def gripper_open(self):
		print("open the gripper")
		self._gripper.open()
		rospy.sleep(1.0)

	def ik_request(self, pose):
		print(pose)
		hdr = Header(stamp=rospy.Time.now(), frame_id='base')
		self._ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
		try:
			resp = self._iksvc(self._ikreq)
		except (rospy.ServiceException, rospy.ROSException) as e:
			print("s")
			rospy.logerr("Service call failed: %s" % (e,))
			return False
		# Check if result valid, and type of seed ultimately used to get solution
		# convert rospy's string representation of uint8[]'s to int's
		resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
		limb_joints = {}
		if (resp_seeds[0] != resp.RESULT_INVALID):
			# seed_str = {
			#             self._ikreq.SEED_USER: 'User Provided Seed',
			#             self._ikreq.SEED_CURRENT: 'Current Joint Angles',
			#             self._ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
			#            }.get(resp_seeds[0], 'None')
			if self._verbose:
				print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}")

			# Format solution into Limb API-compatible dictionary
			limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
			if self._verbose:
				print("IK Joint Solution:\n{0}".format(limb_joints))
				print("------------------")
		else:
			print("no valid joint")
			rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
			return False
		return limb_joints

	def move_to(self, pose, timeout=10.0):
		self._iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK)
		self._ikreq = SolvePositionIKRequest()
		limb_joints = self.ik_request(pose)

		if limb_joints == False:
			return False
		else:
			print('Start Moving to the Pose')
			self._limb.move_to_joint_positions(limb_joints)
			return True
	def get_obs(self):
		rj = self._limb.joint_names()
		joint = self._limb.joint_angle
		joint_state = np.asarray([joint(rj[0]), joint(rj[1]), joint(rj[3]), joint(rj[5]), joint(rj[6])])
		endpoint = np.asarray([robot_pose.x, robot_pose.y, robot_pose.z,
							   robot_orientation.x, robot_orientation.y, robot_orientation.z, robot_orientation.w])
		feature = np.concatenate((joint_state, endpoint))
		obs = {'img': img, 'joint': joint_state, 'endpoint': endpoint}
		return obs

	def set_goal_point(self):
		self.goal_point = Point(x=0.86, y=-0.29, z=-0.00)
		self.goal_orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
		self.goal_pose = Pose(position=self.goal_point, orientation=self.goal_orientation)

	def get_reward(self):
		reward = 0
		distance = np.sqrt((robot_pose.x - self.goal_point.x)**2 + (robot_pose.y - self.goal_point.y)**2 + (robot_pose.z - self.goal_point.z)**2)
		if robot_pose.z < self.goal_point.z:
			reward -= 1
		else:
			reward -= np.sqrt((robot_pose.x - self.goal_point.x)**2 + (robot_pose.y - self.goal_point.y)**2) / (distance + 1e-5)
		reward -= np.tanh(distance)
		reward -= np.sin(3.141592 - abs(self.get_euler()[0]))

		return reward

	def reset(self):
		self.set_goal_point()
		self._limb.move_to_neutral()
		#while self._limb.joint_angle(rj[0]) < 1.0:
		#	self._limb.set_joint_positions({rj[0]: self._limb.joint_angle(rj[0]) + 0.1})
		return self.get_obs()

	def step(self, action):
		joint = self._limb.joint_angle
		rj = self._limb.joint_names()
		joint_state = np.asarray([joint(rj[0]), joint(rj[1]), joint(rj[3]), joint(rj[5]), joint(rj[6])])
		target_joint = joint_state + action
		self._limb.set_joint_positions({
			rj[0]: target_joint[0],
			rj[1]: target_joint[1],
			rj[3]: target_joint[2],
			rj[5]: target_joint[3],
			rj[6]: target_joint[4]
		})
		rospy.sleep(0.1)
		return self.get_obs(), self.get_reward()

	def get_euler(self):
		global robot_orientation
		q_x = robot_orientation.x
		q_y = robot_orientation.y
		q_z = robot_orientation.z
		q_w = robot_orientation.w
		phi = np.arctan2(2.0*(q_x*q_w + q_y*q_z), 1.0 - 2.0*(q_x*q_x+q_y*q_y))
		theta = np.arcsin(2.0*(q_y*q_w - q_x*q_z)) if abs(2.0*(q_y*q_w - q_x*q_z)) < 1.0 \
				else np.copysign(3.141592 / 2, q_y*q_w - q_x*q_z)
		psi = np.arctan2(2.0*(q_z*q_w + q_x*q_y), 1.0 - 2.0*(q_y*q_y + q_z*q_z))
		return [phi, theta, psi]

def pos_callback(data):
	global robot_pose, robot_orientation
	robot_pose = data.pose.position
	robot_orientation = data.pose.orientation

def cam_callback(data):
	bridge = CvBridge()
	global img
	img = bridge.imgmsg_to_cv2(data, "bgr8")
	"""
	np_arr = np.fromstring(data.data, np.uint8)
	image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
	"""
def set_j(limb, joint_name, delta):
	current_position = limb.joint_angle(joint_name)
	print(current_position)
	joint_command = {joint_name: current_position + delta}
	limb.set_joint_positions(joint_command)

def main():


	rospy.init_node("obin_click2pickplace")
	rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, pos_callback)
	rospy.Subscriber("/cameras/right_hand_camera/image", Image, cam_callback, queue_size=1)
	robot = baxter()

	#robot.gripper_open()
	obs = robot.reset()
	print(obs["joint"])
	print(obs["img"].shape)

	for i in range(2):
		for i in range(1000):
			a = 0.1 - 0.2*np.random.sample(robot.act_dim)
			o2, r = robot.step(a)
			print(r)
		robot.reset()
		#cv2.imshow('Frame', o2['img']/255)
	"""
	rj = robot._limb.joint_names()
	while robot._limb.joint_angle(rj[5]) < 1.0:
		robot._limb.set_joint_positions({rj[5]: robot._limb.joint_angle(rj[5]) + 0.1})
	while robot._limb.joint_angle(rj[1]) > 0.0:
		robot._limb.set_joint_positions({rj[1]: robot._limb.joint_angle(rj[1]) - 0.1})
	"""
	#robot._limb.set_joint_positions({rj[1]: 1.0})
	#rospy.sleep(1.0)
	#robot._limb.set_joint_positions({rj[3]: -1.0})
	#rospy.sleep(1.0)
	#robot._limb.set_joint_positions({rj[0]: 1.0})

if __name__ == '__main__':
	sys.exit(main())