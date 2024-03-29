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
import yaml
from pathlib import Path

from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import rospy
from sensor_msgs.msg import Image

NUM_CHANNELS = 3

class ERFNetWrapper:
    def __init__(self, weights_path, class_lut_path, gpu=True, num_threads=-1):
        rospy.loginfo("[ErfNet] Loading weights: " + weights_path)
        rospy.loginfo("[ErfNet] Using GPU: " + str(gpu))
        rospy.loginfo("[ErfNet] CPU Threads: " + str(num_threads))

        self.colorizer_ = Colorize(class_lut_path)

        self.gpu_ = gpu
        if self.gpu_:
            self.device_ = torch.device('cuda')
        else:
            if num_threads > 0:
                torch.set_num_threads(num_threads)
            self.device_ = torch.device('cpu')

        torch.backends.cudnn.benchmark = True
        saved_model_dict = torch.load(weights_path, map_location=self.device_)
        output_size = saved_model_dict[list(saved_model_dict.keys())[-1]].shape[0]
        if output_size != self.colorizer_.num_classes:
            rospy.logwarn(f"[ErfNet] Size of loaded model outputs ({output_size}) and class config "\
                          f"count ({self.colorizer_.num_classes}) do not agree")
        self.model_ = ERFNet(output_size).to(self.device_)

        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        self.model_ = load_my_state_dict(self.model_, saved_model_dict)
        rospy.loginfo("[ErfNet] Model and weights LOADED successfully")
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
            label_color = self.colorizer_(label.unsqueeze(0))
            label_color_np = cv2.resize(label_color.numpy().transpose(1, 2, 0), original_size, 
                                        interpolation = cv2.INTER_NEAREST)
        return label_np, label_color_np

class ERFNetRos:
    def __init__(self):
        self.gen_viz_ = rospy.get_param("~gen_viz", default=False)
        gpu = rospy.get_param("~gpu", default=True)
        num_threads = rospy.get_param("~num_threads", default=-1)
        model_path = rospy.get_param("~model_path", default="../models/model_best.pth")
        world_config_path = rospy.get_param("~world_config_path")

        world_config = yaml.load(open(world_config_path, 'r'), Loader=yaml.CLoader)
        class_path = Path(world_config_path).parent.absolute() / world_config["classes"]

        self.erfnet_ = ERFNetWrapper(model_path, class_path.as_posix(), gpu, num_threads)

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
        rospy.loginfo(f"[ErfNet] Total image callback time: {end_t - start_t} sec")
    
if __name__ == '__main__':
    rospy.init_node("erfnet_ros")
    erfnet_ros = ERFNetRos()
    rospy.spin()
