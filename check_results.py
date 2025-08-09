"""
https://www.makesense.ai/
"""

import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_image(img_name=None, rect_list=[]):
    # define Matplotlib figure and axis
    fig, ax = plt.subplots()
    img = cv2.imread(img_name)
    ax.imshow(img)
    for rct in rect_list:
        rect = Rectangle((rct.x1,rct.y1),
                         rct.x2-rct.x1,
                         rct.y2-rct.y1,
                         linewidth=1,
                         edgecolor='r',
                         facecolor="none")
        ax.add_patch(rect)
    plt.show()

def show_dct(dct, root_dir=None):
    for img,rects in dct.items():
        img = os.path.join(root_dir, img)
        show_image(img_name=img, rect_list=rects)

class Box:
    def __init__ (self,x1=None,y1=None,x2=None,y2=None, iou_val=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.iou_val = 0

    def get_area(self):
        return (self.x2-self.x1)*(self.y2-self.y1)

    def bbx_iou(self,box=None):
        x1 = max(self.x1, box.x1)
        y1 = max(self.y1, box.y1)
        x2 = min(self.x2, box.x2)
        y2 = min(self.y2, box.y2)
        delx,dely = x2-x1, y2-y1
        inter = delx*dely
        if delx<=0 or dely<=0:
            return 0
        ar = self.get_area() + box.get_area()
        iou = inter/(ar - inter)
        return iou

def get_bbox(x1,y1,x2,y2):
    bb = Box()
    bb.x1,bb.y1,bb.x2,bb.y2 = x1,y1,x2,y2
    return bb

def create_dict(filename=None):
    df = pd.read_csv(filename)
    dct = dict()
    N = df.shape[0]
    for k in range(N):
        fn, x1,y1,x2,y2 = df.iloc[k,:]
        bbox = Box(x1=x1, y1=y1,x2=x2,y2=y2)
        lst = dct.get(fn, None)
        if lst == None: lst = []
        lst.append(bbox)
        dct[fn] = lst
    return dct

def process_dcts(pred_dct=None, labs_dct=None, iou_thr=0.5):
    for k,prd_boxes in pred_dct.items():
        boxes = labs_dct.get(k, [])
        for prb in prd_boxes:
            for lbb in boxes:
                iou = prb.bbx_iou(lbb)
                if iou>iou_thr:
                    prb.iou_val += iou
                    lbb.iou_val += iou
    NL, TP = 0,0
    for k, boxes in labs_dct.items():
        NL += len(boxes)
        for bb in boxes:
            if bb.iou_val>=iou_thr:
                TP += bb.iou_val
    TP = TP / NL
    NL, FP = 0,0
    for k, boxes in pred_dct.items():
        NL += len(boxes)
        for bb in boxes:
            if bb.iou_val<=iou_thr:
                FP += 1

    FP = FP/NL
    if FP>0:
        f_score = TP/(TP+FP)
    else:
        f_score = TP
    return f_score

import argparse
if __name__ == "__main__":
    def parse_args():
        ap = argparse.ArgumentParser("Check results")
        ap.add_argument('--pred_csv', type=str, required=True, help="Prediction CSV")
        ap.add_argument('--labs_csv', type=str, required=True, help="Labs CSV")
        ap.add_argument('--img_root_dir', type=str, default="", required=False, help="images root dir")
        args = ap.parse_args()
        return args

    args = parse_args()
    pred_dct = create_dict(args.pred_csv)
    labs_dct = create_dict(args.labs_csv)
    if args.img_root_dir!="":
        show_dct(dct=pred_dct,root_dir=args.img_root_dir)
        #show_dct(dct=labs_dct, root_dir=args.img_root_dir)
        pass
    f_score = process_dcts(pred_dct=pred_dct, labs_dct=labs_dct)
    print(f'f_score:{f_score}')


