#!/usr/bin/env python3
# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import cv2
import torch
import os
import importlib
import time

from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import rospy
from sensor_msgs.msg import Image

NUM_CHANNELS = 3
NUM_CLASSES = 5

class ERFNetWrapper:
    def __init__(self, weights_path, gpu=True):
        print ("Loading weights: " + weights_path)
        print ("Using GPU: " + str(gpu))

        self.gpu_ = gpu
        if self.gpu_:
            self.device_ = torch.device('cuda')
        else:
            self.device_ = torch.device('cpu')

        self.model_ = ERFNet(NUM_CLASSES).to(self.device_)

        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        self.model_ = load_my_state_dict(self.model_, torch.load(weights_path, map_location=self.device_))
        print ("Model and weights LOADED successfully")
        self.model_.eval()

    def infer(self, img_np, gen_viz = False):
        original_size = (img_np.shape[1], img_np.shape[0])
        img_np = cv2.resize(img_np, (640, 400))
        # BGR to RGB
        img_np = np.flip(img_np, axis=2)
        img_np = img_np.transpose((2, 0, 1)).astype(np.float32)/255.
        img = torch.from_numpy(img_np[None, :]).to(self.device_)

        with torch.no_grad():
            outputs = self.model_(img)

        label = outputs[0].max(0)[1].byte().cpu().data
        label_np = cv2.resize(label.numpy(), original_size, interpolation = cv2.INTER_NEAREST)
        label_color_np = None
        if gen_viz:
            label_color = Colorize()(label.unsqueeze(0))
            label_color_np = cv2.resize(label_color.numpy().transpose(1, 2, 0), original_size, 
                                        interpolation = cv2.INTER_NEAREST)
        return label_np, label_color_np

class ERFNetRos:
    def __init__(self):
        self.gen_viz_ = rospy.get_param("~gen_viz", default=False)
        gpu = rospy.get_param("~gpu", default=True)
        self.model_path_ = rospy.get_param("~model_path", default="../models/model_best.pth")

        self.erfnet_ = ERFNetWrapper(self.model_path_, gpu)

        self.image_sub_ = rospy.Subscriber('~image', Image, self.imageCallback, queue_size=100)
        self.label_pub_ = rospy.Publisher('~label', Image, queue_size=10)
        self.label_viz_pub_ =rospy.Publisher('~label_viz', Image, queue_size=1)

    def imageCallback(self, img_msg):
        start_t = time.time()
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        label, label_color = self.erfnet_.infer(img, self.gen_viz_)
        if label_color is not None:
            label_color = np.flip(label_color, axis=2)
            label_viz_msg = Image()
            label_viz_msg.header = img_msg.header
            label_viz_msg.encoding = "bgr8"
            label_viz_msg.height = label_color.shape[0]
            label_viz_msg.width = label_color.shape[1]
            label_viz_msg.step = label_viz_msg.width * 3
            label_viz_msg.data = label_color.tobytes()
            self.label_viz_pub_.publish(label_viz_msg)

        label_msg = Image()
        label_msg.header = img_msg.header
        label_msg.encoding = "mono8"
        label_msg.height = label.shape[0]
        label_msg.width = label.shape[1]
        label_msg.step = label_msg.width
        label_msg.data = label.tobytes()
        self.label_pub_.publish(label_msg)
        end_t = time.time()
        rospy.loginfo(f"ErfNet total image callback time: {end_t - start_t} sec")
    
if __name__ == '__main__':
    rospy.init_node("erfnet_ros")
    erfnet_ros = ERFNetRos()
    rospy.spin()
