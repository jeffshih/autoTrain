import csv
import json
import sys
import os
sys.path.append('/root/pva-faster-rcnn/lib')
sys.path.append('/root/pva-faster-rcnn/lib/datasets')
import time
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import json
from os import listdir
from os.path import isfile, join
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import glob
import cv2
from datasets.config import CLASS_SETS
from natsort import natsorted




def load_meta(meta_path):
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
       
        meta = {"format":"jpg"}
        meta["train"] = {"start":None, "end":None, "stride":1, "sets":[0]}
        meta["test"] = {"start":None, "end":None, "stride":30, "sets":[1]}
        print("Meta data path: {} does not exist. Use Default meta data".format(meta_path))
    return meta
 





class IMDBGroup(imdb):
    
    
    def _check_consistency(self):
        
        for dataset in self._datasets[1:]:
            assert self._datasets[0]._classes == dataset._classes, \
            "The class set are inconsistent.  {}/{}  and {}/{}".format(self._datasets[0].name,\
                                                                       self._datasets[0]._classes, dataset.name, dataset._classes)
    
    def _get_img_paths(self):
        
        
        img_paths = []
        
        for dataset in self._datasets:
            for i in range(len(dataset._image_index)):
                img_path = dataset.image_path_at(i)
                img_paths.append(img_path)
            
        return img_paths   
            
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]
    
    
    
    def gt_roidb(self):
        
        gt_roidb = []
        for dataset in self._datasets:
            gt_roidb += dataset.gt_roidb()
        return gt_roidb

    
    def __init__(self, datasets):
        self._datasets = datasets
        self._check_consistency()
        self._classes = self._datasets[0]._classes
        name = " ".join([dataset.name for dataset in datasets])
        
        imdb.__init__(self,'IMDB Groups:{}'.format(name))
      

        self._image_index = self._get_img_paths()
        
        
        
        




class openImageData(imdb):
    
    
            
    def loadMapper(self,mapperPath,mapperList):
        mapper = {}
        reverseMapper = {}
        f = open(mapperPath,'r')
        for i in csv.reader(f):
            key = i[0]
	    val = i[1]
            reverseMapper[val]=key
	    if mapperList.has_key(i[1]):
                val = mapperList.get(i[1])
            else:
                val = i[1]
            mapper[key] = val
        f.close()
        return mapper,reverseMapper  
    
    def getAnnotation(self,labelList,mapperList,sets='train'):
        target_imgs = set()
        mapperPath = '/root/data/data-openImages_v4/class-descriptions-boxable.csv'
        mapper,reverseMapper = self.loadMapper(mapperPath,mapperList)
        method = ['freeform ','xclick']
        bboxGTPath = '/root/data/data-openImages_v4/{}-annotations-transformed.csv'.format(sets)
        seq = [reverseMapper.get(i) for i in labelList]
        f = open(bboxGTPath, 'r')
        annotations = {}
        mappedClass = {}
        for row in csv.reader(f):
            if row[1] not in method:
                continue
            if row[2] not in seq:
                continue
	    if row[10] == '1':
		continue
            if os.path.isfile(os.path.join('/root/data/data-openImages_v4/{}'.format(sets),row[0]+'.jpg')):
                if annotations.has_key(row[0]):
                    annotations[row[0]] += [row[2:]]
                    mappedClass[row[0]] += [[mapper.get(row[2])]+row[3:]]
                    target_imgs.add(row[0])
                else:
                    annotations[row[0]] = [row[2:]]
                    mappedClass[row[0]] = [[mapper.get(row[2])]+row[3:]]
                    target_imgs.add(row[0])
        target_imgs = list(target_imgs) 
        print("Total: {} images".format(len(target_imgs)))
        f.close()
        return annotations,mappedClass,target_imgs
    
    


    def __init__(self, datasetName,cls_order,cls_mapper):
        
        name="openImages_v4"
        #FOR DEBUGGING
        self.debugging = True
        


        imdb.__init__(self,name)
        
        self._data_path = '/root/data/data-openImages_v4'
        
        #CLS_mapper,labelList =  self.parseConfig()
        print cls_order
        self._classes = cls_order
        print cls_mapper
        self.CLS_mapper = cls_mapper
        
        namedAnnotation,annotation,target_imgs = self.getAnnotation(self._classes,self.CLS_mapper)   
        self._annotation = annotation       
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        
        self.original_classes = self.get_original_classes()
        
        meta_data_path = os.path.join(self._data_path, "meta.json")         
        self._meta = load_meta(meta_data_path)
        self._image_ext = self._meta["format"]    
        self._image_ext = '.jpg'
        self._image_index = target_imgs
        print "images set loaded"
    
        
    def get_original_classes(self):
        original_classes = set()
        for bboxes in self._annotation.values():
            original_classes.add(bboxes[0][0])
        return original_classes    
    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'train', index+self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions 

        This function loads/saves from/to a cache file to speed up future calls.
        """

        gt_roidb = []
        for index in self.image_index:         
            boxes = self._load_boxes(index)
            gt_roidb.append(boxes)
        return gt_roidb


    def rpn_roidb(self):
       
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)


        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

  
    
    #Assign negtaive example to __background__ as whole image
    def _load_boxes(self, index):
     
        bboxes = self._annotation.get(index,{})
       # print bboxes
        #print(frame, "Before", bboxes)
        bboxes = [bbox for bbox in bboxes if bbox[8]!='1']
        num_objs = len(bboxes)
            

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        #Becareful about the coordinate format
        # Load object bounding boxes into a data frame.
  
        #print(bboxes)
        # This is possitive example
        for ix, bbox in enumerate(bboxes):
            #print(bbox)
         
            
            x1 = float(bbox[2])
            y1 = float(bbox[4])
            x2 = float(bbox[3])
            y2 = float(bbox[5])
            label = bbox[0]
            #print(label, self.CLS_mapper)
            if label in self.CLS_mapper:
                label = self.CLS_mapper[label]
                if label == '__background__':
                    continue
            cls = self._class_to_ind[label]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls  
            overlaps[ix, cls] = 1.0
            
            
             
            seg_areas[ix] = (x2 - x1) * (y2 - y1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}



if __name__ == '__main__':
    from datasets.openImage import openImageData
    from datasets.openImage import IMDBGroup
    
    name_a = "chruch_street"
    
    #B = openImageData("Monkey", class_set_name)
    print "A classes are:",A._classes
    #imdb_group = IMDBGroup([A,B])
    
    
    

