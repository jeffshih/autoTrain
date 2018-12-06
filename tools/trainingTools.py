from argparse import ArgumentParser
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
import json
import csv

from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(loader=FileSystemLoader('/'))
train_template = "root/pva-faster-rcnn/models/pvanet/lite/template_train.prototxt" 
test_template = "root/pva-faster-rcnn/models/pvanet/lite/template_test.prototxt" 
solver_template = "root/pva-faster-rcnn/models/pvanet/lite/template_solver.prototxt" 
bbox_pred_name = "bbox_pred-coco"
cls_score_name = "cls_score-coco"


def combined_roidb(imdb):
    def get_roidb(imdb):
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    
    roidb = get_roidb(imdb)
  
    return imdb, roidb 

def prepare_data(model_name,classPath,cls_order):
     #Set the training configuration first
    cfg_path="models/pvanet/lite/train.yml"
    cfg_from_file(cfg_path)
    
    datasets=openImageData(model_name,classPath,cls_order)
    #imdb_group = IMDBGroup(datasets)
    imdb, roidb = combined_roidb(datasets)
    
    
    total_len = float(len(datasets.gt_roidb()))

    #Show the dataset percentage in the whole composition
    img_nums = len(datasets.gt_roidb())   
    print(datasets.name, img_nums,  "{0:.2f}%".format(img_nums/total_len * 100))
    
    return roidb

def finetune(net_params, roidb, GPU_ID=1):

   
    #solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix = net_params
    solver, train_pt, caffenet, max_iters, output_dir, output_prefix = net_params
    

        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print 'Trained model will be saved to `{:s}`'.format(output_dir)
    
    
    
    
    
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)  
    
    
    train_net(solver, roidb, output_dir, output_prefix,
              pretrained_model=caffenet, max_iters=max_iters, bbox_pred_name="bbox_pred-coco")



def generate_prototxt(template_path, output_name, num_classes, bbox_pred_name, cls_score_name):
    output = env.get_template(template_path).render(num_classes= num_classes,\
                                                 bbox_pred_name=bbox_pred_name, cls_score_name=cls_score_name)

    w = open(output_name, 'w')
    w.write(output)
    w.close()

def generate_solver(template_path, output_name,train_path, num_classes, bbox_pred_name, cls_score_name):
    output = env.get_template(template_path).render(num_classes= num_classes,\
                                                 bbox_pred_name=bbox_pred_name, cls_score_name=cls_score_name,train_path=train_path)
    w = open(output_name, 'w')
    w.write(output)
    w.close()
    
def generate(train_template,test_template,solver_template,num_classes,bbox_pred_name,cls_score_name,output_prefix):
    #Generating Training Prototxt
    train_path = "/root/pva-faster-rcnn/models/pvanet/lite/{}_train.prototxt".format(output_prefix)
    generate_prototxt(train_template, train_path, num_classes, bbox_pred_name, cls_score_name)

    #Generating Testing Prototxt
    test_path = "/root/pva-faster-rcnn/models/pvanet/lite/{}_test.prototxt".format(output_prefix)
    generate_prototxt(test_template, test_path, num_classes, bbox_pred_name, cls_score_name)

    #Generating Solver Prototxt
    #Generating Training Prototxt
    solver_path = "/root/pva-faster-rcnn/models/pvanet/lite/{}_solver.prototxt".format(output_prefix)
    generate_solver(solver_template, solver_path,train_path, num_classes, bbox_pred_name, cls_score_name)
    
    
    
    
if __name__ == '__main__':
    

    parser = ArgumentParser()
    parser.add_argument("-i", "--index",help="class index mapper",dest="mapper",default="{\"cls_mapper\": {\"Kitchen knife\":\"Knife\",\"Dagger\":\"Knife\",\"Shotgun\":\"Handgun\",\"Rifle\":\"Handgun\"}, \"cls_order\": [\"__background__\",\"Shotgun\",\"Rifle\",\"Handgun\",\"Knife\",\"Kitchen knife\",\"Dagger\"],\"gpu_id\":\"0\"}")
    parser.add_argument("-n", "--name", help="model name", dest="name", default="default")
    parser.add_argument("-c", "--configPath",help="config path",dest="path",default=None)

    
    args = parser.parse_args()
    
    if (args.path == None):
	config = json.loads(args.mapper)
    else:
	print args.path
	jsonInput = open(args.path,'rb')
	print jsonInput
	config = json.load(jsonInput)

    model_name = args.name
    gpu_id = config.get("gpu_id")
    print "gpu_id = ",gpu_id

     
    #num_classes = len(a["labelList"])
    #cls_order = a["labelList"]
    cls_order = config.get("cls_order")
    cls_mapper = config.get("cls_mapper")
    roidb = prepare_data(model_name,cls_order,cls_mapper) 
    tempClass = []
    for cls in cls_order:
	if cls_mapper.has_key(cls):
		if cls_mapper.get(cls) not in tempClass:
	           tempClass.append(cls_mapper.get(cls))
   	elif cls not in tempClass:
		tempClass.append(cls) 
    print "replaced class = ",tempClass
    print "num of total classes = ",len(tempClass)
    num_classes = len(tempClass)-1
    
#    num_classes = len(cls_order)
    print "total_class = ",num_classes
    output_prefix=model_name
    
    
    
    solver = "/root/pva-faster-rcnn/models/pvanet/lite/{}_solver.prototxt".format(output_prefix)
    train_pt = "/root/pva-faster-rcnn/models/pvanet/lite/{}_train.prototxt".format(output_prefix)
    caffenet = "/root/pva-faster-rcnn/models/pvanet/pretrained/pva9.1_preAct_train_iter_1900000.caffemodel" 
    #caffenet = "/root/pva-faster-rcnn/models/output/human0806/human0806_iter_198000.caffemodel"
    #caffenet = "/root/pva-faster-rcnn/models/output/weaponModel_0720/weaponModel_0720_iter_48000.caffemodel"    
    #roidb = prepare_data(model_name,cls_order,cls_mapper)
    
    generate(train_template,test_template,solver_template,num_classes,bbox_pred_name,cls_score_name,output_prefix)
    bbox_pred_name = "bbox_pred-coco"
    #The ouput directory and prefix for snapshots
    output_dir = "/root/pva-faster-rcnn/models/output/{}".format(output_prefix)    
    #The maximum iterations is controlled in here instead of in solver
    max_iters = 5 * 100000       
    #net_params = (solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix)
    net_params = (solver, train_pt, caffenet, max_iters, output_dir, output_prefix)
    
    GPU_ID = int(gpu_id)
    finetune(net_params, roidb, GPU_ID)


