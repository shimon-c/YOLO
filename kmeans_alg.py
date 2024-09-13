import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import random

def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

def mat_dist(p1,m2):
    diff = p1-m2
    diff = diff * diff
    dist = np.sum(diff,axis=1)
    dist = np.sqrt(dist)
    return dist

class KMeansEM:
    def __init__(self, num_clusters=9):
        self.num_clusters = num_clusters
        self.pnts_assign = []
        self.clusters = None

    # Implementing E step
    def assign_clusters(self, X, clusters):
        num_changes = 0
        KK = clusters.shape[0]
        centers = np.zeros((KK, clusters[0]['center'].shape[0]))
        for i in range(KK):
            centers[i,...] = clusters[i]['center']
        for idx in range(X.shape[0]):
            dist = []

            curr_x = X[idx]
            KK = clusters.shape[0]
            # for i in range(KK):
            #     dis = distance(curr_x, clusters[i]['center'])
            #     dist.append(dis)
            dist = mat_dist(curr_x, centers)
            curr_cluster = np.argmin(dist)
            #c1 = np.argmin(dist1)
            if self.pnts_assign[idx] != curr_cluster:
                num_changes += 1
            self.pnts_assign[idx] = curr_cluster
            clusters[curr_cluster]['points'].append(curr_x)
        return clusters,num_changes


    # Implementing the M-Step
    def update_clusters(self, X, clusters):
        KK = clusters.shape[0]
        for i in range(KK):
            points = np.array(clusters[i]['points'])
            if points.shape[0] > 0:
                new_center = points.mean(axis=0)
                clusters[i]['center'] = new_center
                clusters[i]['size'] = len(points)
                clusters[i]['points'] = []
        return clusters

    # Each row is an example
    def fit(self, data=None, num_iters=100):
        data_np = np.array(data)
        N = data_np.shape[0]
        clusters_ids = random.sample([k for k in range(N)], self.num_clusters)
        clusters_init = data_np[clusters_ids,:]
        clusters = [dict() for k in range(self.num_clusters)]
        clusters = np.array(clusters)
        self.pnts_assign = np.array([-1 for k in range(N)])
        for k in range(self.num_clusters):
            clusters[k]['points'] = []
            clusters[k]['center'] = clusters_init[k]
        num_changes = -1
        for iter in tqdm(range(num_iters),  desc="Kmeans fit:"):
            clusters, num_changes = self.assign_clusters(data_np, clusters)
            print(f"iter:{iter},num_changes:{num_changes}")
            if num_changes <= 0:
                break
            self.update_clusters(data_np, clusters)
        self.clusters = clusters

    def select(self, K=9):
        NK = self.clusters.shape[0]
        clsz = np.zeros((self.clusters.shape[0],))
        for k in range(NK):
            clsz[k] = self.clusters[k]['size']
        ids = np.argsort(clsz, )
        ids = ids[::-1]
        self.clusters = self.clusters[ids[:K]]

    def get_clusters(self):
        NK = self.clusters.shape[0]
        dim = self.clusters[0]['center'].shape[0]
        clts = np.zeros((NK,dim))
        for k in range(NK):
            clts[k,:] = self.clusters[k]['center']
        clts = clts[clts[:, 1].argsort()]
        return clts


class KMeanAlg:
    def __init__(self, csv_file:str='', label_dir=''):
        self.annotations = pd.read_csv(csv_file)
        self.label_dir = label_dir
        self.boxes = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        index = index % len(self.annotations)
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        wh_boxes = []
        for box in bboxes:
            x, y, width, height, class_label = box
            wh_boxes.append((width,height))
        return wh_boxes

    def get_dataset(self):
        L = len(self)
        data = []
        for k in tqdm(range(L), desc="get_dataset"):
            boxes = self[k]
            data.extend(boxes)
        return data

    def get_kmean(self, K=9, run_keamns_em=True):
        data = self.get_dataset()
        if run_keamns_em:
            kmeans = KMeans(n_clusters=K)
            kmeans = KMeansEM(num_clusters=30)
            kmeans.fit(data)
            kmeans.select(K=K)
            clts = kmeans.get_clusters()
            print(f'KmeanEM clusters:\n{clts}')
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(data)
        boxes = kmeans.cluster_centers_
        boxes = boxes[boxes[:, 1].argsort()]
        print(f"SKlearn boxes:\n{boxes}")
        self.boxes = boxes
        return boxes

import argparse
def parse_args():
    ap = argparse.ArgumentParser("Kmeans for aspect ratios")
    ap.add_argument('--annotation_file', type=str,required=True, help="Full path of annotation file")
    ap.add_argument('--labels_dir', type=str, required=True, help="Full path of labels dir")
    ap.add_argument('--k_ratios', type=int, default=9, help="Number of aspect ratios")
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    km = KMeanAlg(csv_file=args.annotation_file, label_dir=args.labels_dir)
    boxes = km.get_kmean(K=args.k_ratios)

# cmd: pyhton --annotation_file="D:\PASCAL_VOC\train.csv" --labels_dir=D:\PASCAL_VOC\labels