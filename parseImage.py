import os
import csv
import cv2
import sys
imgSize = []
cnt = 0
totalCnt = 1743042


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)+'\b')
    # Print New Line on Complete
    if iteration == total: 
        print()

for root,subdir,fnames in os.walk('/root/data/data-openImages_v4/validation'):
    for fname in fnames:
        path = os.path.join(root,fname)
        filename = fname.split('.')[0]
        if path.endswith('jpg'):
            img = cv2.imread(path)
            imgSize.append([fname,img.shape[0:2]])
        else:
            print "{} is broke".format(path)
#	cnt+=1
#	printProgressBar(cnt+1,totalCnt,prefix='Progress',suffix='Complete')
	
with open("validationImgSizeMapper.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(imgSize)

