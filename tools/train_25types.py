"""
RC4
COCO 2014 train + coco 2014 val + Vatic[A1Highwayday, B2HighwayNight,  airports, airports2] + the SET00 part [luggage, luggage2] of on Training

Removing Yu Da Campus data from RC2 since the backpack, handnag, and suitcase are missing.

"""



"""This is an example of fine tuning PVA-NET through IMDB moudle that combines different dataset together
Most of the codes are modified from Faster R-CNN

"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import datasets.imdb
from datasets.coco import coco
from datasets.openImage import openImageData,IMDBGroup
from datasets.pascal_voc_new import pascal_voc
import caffe
import argparse
import pprint
import numpy as np
import sys
import os


def combined_roidb(imdb):
    def get_roidb(imdb):
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    
    roidb = get_roidb(imdb)
  
    return imdb, roidb 



def prepare_data():
     #Set the training configuration first
    cfg_path="models/pvanet/lite/train.yml"
    cfg_from_file(cfg_path)
    
    """
     1. PREPARING DATASET
    """
    


    #Firstly, prepare the dataset for fine-tuning 
    #Different kind of dataset is wrapped by the IMDB class, originally designed by Ross Girshick    
    
    #You need to put coco data directory(soft-link works as well) under the PVA-NET directory
    #COCO IMDB needs two parameter: data-split and year    
 
   #coco_train = coco("train", "2014")   
    #coco_val = coco("val", "2014")    
    
    #Fetch the classes of coco dataet, this will be useful in the following section

    #classes = coco_val._classes
    #Next, we import the VOC dataset via pascal_voc wrapper
    #Since VOC and COCO data have different naming among classes, a naming mapper is needed to unify the class names
 
    #mapper = {"tvmonitor":"tv", "sofa":"couch", "aeroplane":"airplane",
#              "motorbike":"motorcycle", "diningtable":"dining table", "pottedplant":"potted plant"}
    
    
   

    #Finnaly, let's wrap datasets from Vatic.
    #A vatic dataset directory should be located under ~/data/ directory in the naming of data-*
    #For example: ~/data/data-YuDa,  ~/data/data-A1HighwayDay
    #vatic_names = ["A1HighwayDay", "B2HighwayNight", "airport", "airport2"]    
    
#    mapper = {"van":"car", "trailer-head":"truck",\
#              "sedan/suv":"car", "scooter":"motorcycle", "bike":"bicycle"}    
    
#    vatics = [VaticData(vatic_name, classes, CLS_mapper=mapper, train_split="all") for vatic_name in vatic_names]
    
    #openImages = [openImageData(dataName, classes, CLS_mapper=mapperList, train_split="train") for dataName in dataNames]
    
    
    #Combine all the IMDBs into one single IMDB for training
#    datasets = vatics + [coco_train, coco_val]      
    datasets=openImageData("test","/root/pva-faster-rcnn/smallSample.json")
    #imdb_group = IMDBGroup(datasets)
    imdb, roidb = combined_roidb(datasets)
    
    
    total_len = float(len(datasets.gt_roidb()))

    #Show the dataset percentage in the whole composition
    img_nums = len(datasets.gt_roidb())   
    print(datasets.name, img_nums,  "{0:.2f}%".format(img_nums/total_len * 100))
    
    return roidb


def finetune(net_params, roidb, GPU_ID=1):

   
    solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix = net_params
    
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print 'Trained model will be saved to `{:s}`'.format(output_dir)
    
    
    
    
    
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)  
    
    
    train_net(solver, roidb, output_dir, output_prefix,
              pretrained_model=caffenet, max_iters=max_iters, bbox_pred_name="bbox_pred-coco")
    
    
    
    





if __name__ == '__main__':
    
    
    #Prepare roidb
    roidb = prepare_data()
    
   
    
    
       
    # Set each training parameter    
    solver = "/root/pva-faster-rcnn/models/pvanet/lite/three_solver.prototxt"
    train_pt = "/root/pva-faster-rcnn/models/pvanet/lite/three_train.prototxt"
    caffenet = "/root/pva-faster-rcnn/models/pvanet/lite/test.model"
    
    #The bbox_pred_name is used to specify the new name of bbox_pred layer in the modified prototxt. bbox_pred layer is handeled differentlly in the snapshooting procedure for the purpose of bbox normalization. In order to prevent sanpshotting the un-tuned bbox_pred layer, we need to specify the new name.  
    bbox_pred_name = "bbox_pred-coco"
    #The ouput directory and prefix for snapshots
    output_dir = "/root/pva-faster-rcnn/models/rc/25types"
    output_prefix = "25types"    
    #The maximum iterations is controlled in here instead of in solver
    max_iters = 100 * 10000       
    net_params = (solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix)
    
    GPU_ID = 1
    #Start to finetune
    finetune(net_params, roidb, GPU_ID)
    
