#!/usr/bin/env python3

import torch
from erfnet import ERFNet

NUM_CLASSES = 7
model_ = torch.nn.DataParallel(ERFNet(NUM_CLASSES)).cuda()

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        own_state[name].copy_(param)
    return model

model_ = load_my_state_dict(model_, torch.load('../models/model_best.pth'))

torch.save(model_.module.state_dict(), 'model_generic.pth')
