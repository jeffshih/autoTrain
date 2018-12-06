import _init_paths

import os
import sys
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import caffe
import os
import cv2
from utils.timer import Timer
import numpy as np
import time

caffe.set_mode_gpu()
caffe.set_device(3)


#cfg_from_file("/tmp/test/submit_1019.yml")
cfg_from_file("/tmp/test/submit_0716.yml")
prototxt = "/tmp/test/weaponModel_test.prototxt"
caffemodel = "/tmp/test/weaponModel_iter_6000.caffemodel"
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

im = cv2.imread("/tmp/test/test.jpg")

_t = {'im_preproc': Timer(), 'im_net': Timer(),
      'im_postproc': Timer(), 'misc': Timer()}

scores, boxes = im_detect(net, im, _t)

for i in range(10):
    _s = time.time()
    scores, boxes = im_detect(net, im, _t)
    _e = time.time()
    print "time: %s" % (_e-_s)
    time.sleep(1)
