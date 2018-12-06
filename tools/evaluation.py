import sys
import os
sys.path.append('/root/pva-faster-rcnn/lib')
sys.path.append('/root/pva-faster-rcnn/lib/datasets')
sys.path.append('/root/pva-faster-rcnn/tools')
import glob
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect,get_layer_name
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, cv2
import argparse
import json
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.config import CLASS_SETS
from datasets.vatic_hierarchy import VaticData, IMDBGroup
from datasets.openImage import openImageData
#from datasets.vatic import VaticData, IMDBGroup
import random
from scipy.misc import imread
import csv
from argparse import ArgumentParser

MINOVERLAP = 0.2


class AINVReval():
    
    def detect(self,CLASSES,net,img,threshold):
        im = cv2.imread(img)
        NMS_THRESH = 0.3
        _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
        scores,boxes = im_detect(net,im,_t)
        res = [] 
        for cls_ind, cls in enumerate(CLASSES[1:]):
            #print cls_ind
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            thresh = 0
            inds = np.where(dets[:, -1] > thresh)[0]
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]*100
                #Fix bug 6
                x = bbox[0]
                y = bbox[1]
                width = bbox[2] - bbox[0]
                height =  bbox[3] - bbox[1]
                label = cls
                socre = bbox[-1] * 100
                if score > threshold:
                    res.append([img, x, y, width, height, score, label])

        return res

    def processClass(self,cls_order,cls_mapper):
        tempCls=[]
        for i in cls_order:
            if cls_mapper.has_key(i):
                if cls_mapper.get(i) not in tempCls:
                    tempCls.append(cls_mapper.get(i))
            elif i not in tempCls:
                tempCls.append(i)
        return tempCls

    def calVocAp(self,rec,prec):
        rec.insert(0,0.0)
        rec.append(1.0)
        mrec = rec[:]
        prec.insert(0,0.0)
        prec.append(0.0)
        mpre = prec[:]
        for i in range(len(mpre)-2,-1,-1):
            mpre[i] = max(mpre[i],mpre[i+1])
        iList = []
        for i in range(1,len(mrec)):
            if mrec[i] != mrec[i-1]:
                iList.append(i)

        ap = 0.0
        for i in iList:
            ap+=((mrec[i]-mrec[i-1])*mpre[i])
        return ap,mrec,mpre



    def getDetectionData(self,modelName,prototxtName,cls_order,cls_mapper,gpu_id,testSets,datasetName="openImages_v4"):
        imgIdx = []
        GT = []
        CLASSES = self.processClass(cls_order,cls_mapper)
        name = "eval"
        for target in testSets:
            dataset = openImageData(name,cls_order,cls_mapper,sets=target)
            for i in dataset._image_index:
                target_img = "/root/data/data-{}/{}/{}.jpg".format(datasetName,target,i)
                imgIdx += [target_img]
                label = dataset._load_boxes(i)['gt_classes']
                bbox = dataset._load_boxes(i)['boxes']
                for j,k in zip(label,bbox):
                    GT.append(([target_img,k[0],k[1],k[2],k[3],CLASSES[j]]))
        GTdata = {}
        gtCounterPerCls = {}
        for re in GT:
            if gtCounterPerCls.has_key(re[5]):
                gtCounterPerCls[re[5]]+=1
            else:
                gtCounterPerCls[re[5]] = 1
            hit = {"label":re[5],"bbox":[x for x in re[1:5]],"used":False}
            if GTdata.has_key(re[0]):
                GTdata.get(re[0]).append(hit)
            else:
                GTdata[re[0]] = []
                GTdata.get(re[0]).append(hit)   
        weightPath = modelName#"/root/pva-faster-rcnn/models/output/{}/{}_iter_{}.caffemodel".format(modelName,modelName,str(iteration))
        modelPath = prototxtName#"/root/pva-faster-rcnn/models/pvanet/lite/{}_test.prototxt".format(modelName)
        print weightPath
        print modelPath
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        cfg_from_file("models/pvanet/cfgs/submit_0716.yml")
        cfg.GPU_ID = gpu_id
        det = {}
        totalImg = len(imgIdx)                       
        if not os.path.isfile(weightPath):
            raise IOError(('Caffemodel: {:s} not found').format(weightPath))
        net = caffe.Net(modelPath, weightPath, caffe.TEST)
        print '\n\nLoaded network {:s}'.format(modelPath)
        print "Total testing images: {}".format(len(imgIdx))
        for idx,targetImg in enumerate(imgIdx):
            timer = Timer()
            timer.tic()
            _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
            res = self.detect(CLASSES,net,targetImg,15)
            timer.toc()
            print "Processing: {}/{},time usage is {:.3f}s".format(idx,totalImg,timer.average_time)
            for re in res:
                hit = {"confidence":re[5],"fileName":re[0],"bbox":[x for x in re[1:5]]}
                if det.has_key(re[6]):
                    det.get(re[6]).append(hit)
                else:
                    det[re[6]] = []
                    det.get(re[6]).append(hit)             
        for i in CLASSES:
            if det.has_key(i) == False:
                det[i] = []
        return det,GTdata,gtCounterPerCls

    def calculate(self,CLASSES,Mapper,res ,GTdata,gtCounterPerCls):
        cls_order = self.processClass(CLASSES,Mapper)
        gtClasses = cls_order
        gtClasses = sorted(cls_order)
        n_classes = len(gtClasses)
        sumAP = 0.0
        result = {}
        countTP = {}
        for clsIdx,clsName in enumerate(gtClasses):
            countTP[clsName] = 0
            #GTdata   = GTdata
            if clsName == '__background__':
                continue
            #print clsName
            predData = res.get(clsName)
            nd = len(predData)
            tp = [0]*nd
            fp = [0]*nd
            for idx, pred in enumerate(predData):
                fileName = pred["fileName"]
                ovmax = -1
                gtMatch = -1
                gt = GTdata.get(fileName)
                #print gt
                bb = pred['bbox']
                for obj in gt:
                    if obj["label"]== clsName:
                        bbgt = [float(x) for x in obj["bbox"]]
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2]-bi[0]+1
                        ih = bi[3]-bi[1]+1
                        if iw>0 and ih>0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw*ih/ua
                            if ov > ovmax:
                                ovmax = ov
                                gtMatch = obj
                minOverlap = MINOVERLAP
                if ovmax>minOverlap:
                    if not bool(gtMatch["used"]):
                        tp[idx]=1
                        gtMatch["used"]=True
                        countTP[clsName]+=1
                    else:
                        fp[idx]=1
                else:
                    fp[idx]=1
            cumsum = 0
            for idx,val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx,val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]
            for idx,val in enumerate(tp):
                rec[idx] = float(tp[idx])/gtCounterPerCls[clsName]
            prec = tp[:]
            for idx,val in enumerate(tp):
                prec[idx] = float(tp[idx])/(fp[idx]+tp[idx])
            ap,mrec,mprec = self.calVocAp(rec,prec)
            #print clsName, ap
            sumAP += ap
            text = "{0:.4f}%".format(ap*100) + " = " + clsName + " AP  " 
            roundedPrec = [ '%.4f' % elem for elem in prec ]
            averagePrec = sum(prec)/len(prec)
            roundedRec = [ '%.4f' % elem for elem in rec ]
            averageRec = sum(rec)/len(rec)
            print "{} average Precision, Recall is {},{}".format(clsName,averagePrec,averageRec)
            #resultsFile.write(text + "\n Precision: " + str(averagePrec) + "\n Recall   :" + str(averageRec) + "\n\n")
            result[clsName] = {"recall":averageRec,"precision":averagePrec}
        mAP = sumAP / n_classes
        print "mean Average Precision is {}".format(mAP)
        return result

    def evaluate(self,clsOrder,clsMapper,model,prototxt,gpu_id): 
        res,GTdata,gtCounterPerCls = self.getDetectionData(model,prototxt,clsOrder,clsMapper,1,["validation"])
        return self.calculate(clsOrder,clsMapper,res,GTdata,gtCounterPerCls)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--class",help="class index mapper",dest="cls",default="{\"cls_mapper\": {\"Kitchen knife\":\"Knife\",\"Dagger\":\"Knife\",\"Shotgun\":\"Handgun\",\"Rifle\":\"Handgun\"}, \"cls_order\": [\"__background__\",\"Shotgun\",\"Rifle\",\"Handgun\",\"Knife\",\"Kitchen knife\",\"Dagger\"]}")
    parser.add_argument("-m","--model",help="path to caffemodel",dest="model",default="/root/pva-faster-rcnn/models/output/weapon0806/weapon0806_iter_220000.caffemodel")
    parser.add_argument("-p","--prototxt",help="path to prototxt",dest="prototxt",default="/root/pva-faster-rcnn/models/pvanet/lite/weapon0806_test.prototxt")
    parser.add_argument("-g","--gpu_id",help="GPU ID",dest="gpu_id",default="1")
    
    args = parser.parse_args()
    cls = json.loads(args.cls)
    clsOrder = cls.get("cls_order")
    clsMapper = cls.get("cls_mapper")
    model = args.model
    prototxt = args.prototxt
    gpu_id = args.gpu_id
    evaluation = AINVReval()
    res = evaluation.evaluate(clsOrder,clsMapper,model,prototxt,gpu_id)