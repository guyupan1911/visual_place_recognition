import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import kapture
import kapture.io.csv as csv
from kapture.core.flatten import flatten
from kapture.io.records import get_image_fullpath

import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import quaternion
from PIL import Image
from tqdm import tqdm

query_root_path = '/autox-dl/localization/guyu/dataset/yuemeite/query/kapture_data'
db_root_path = '/autox-dl/localization/guyu/dataset/yuemeite/mapping/kapture_data'


def input_transform():
    return transforms.Compose([
        transforms.Resize((640,640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_test_set():
    dataset = WholeDataset(transform=input_transform())
    return ImagesFromList(dataset.qImages[dataset.qIdx], input_transform()), \
        ImagesFromList(dataset.dbImages, input_transform()), dataset.nonNegIdx

def get_train_set():
    dataset = WholeDataset(transform=input_transform(), neg_threshold=50)
    return dataset

class ImagesFromList(data.Dataset):
	def __init__(self, images, transform):

	    self.images = images
	    self.transform = transform

	def __len__(self):
	    return len(self.images)

	def __getitem__(self, idx):

		img = [Image.open(im) for im in self.images[idx].split(",")]
		img = [self.transform(im) for im in img]

		if len(img) == 1:
			img = img[0]

		return img, idx

class WholeDataset(data.Dataset):
    def __init__(self, transform, margin=0.1, nNeg=5, pos_threshold=10, neg_threshold=25):
        self.transform = transform
        self.margin = margin
        self.nNeg = nNeg
        # 1. read kapture data
        db_kapture_data = csv.kapture_from_dir(db_root_path)

        query_kapture_data = csv.kapture_from_dir(query_root_path)
        
        db_image_list = flatten(db_kapture_data.records_camera)
        query_image_list = flatten(query_kapture_data.records_camera)

        ## read specified camera id images
        specified_image_sensor_id = ['right_0_n_6mm']

        db_image_names = [(ts, get_image_fullpath(db_root_path, name)) for ts,sensor_id,name in db_image_list if sensor_id in specified_image_sensor_id]
        query_image_names = [(ts, get_image_fullpath(query_root_path, name)) for ts,sensor_id,name in query_image_list if sensor_id in specified_image_sensor_id]
        
        ## read trajectories
        db_trajectories = [db_kapture_data.trajectories.intermediate_pose(item[0],'car',1e9) for item in db_image_names]
        query_trajectories = [query_kapture_data.trajectories.intermediate_pose(item[0],'car',1e9) for item in query_image_names]
        
        self.utmDB = np.zeros((len(db_trajectories), 7))
        for i, pose in enumerate(db_trajectories):
            self.utmDB[i,0:4] = np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z])
            self.utmDB[i,4:7] = pose.t.reshape(1,3)
        
        self.utmQ = np.zeros((len(query_trajectories), 7))
        for i, pose in enumerate(query_trajectories):
            self.utmQ[i,0:4] = np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z])
            self.utmQ[i,4:7] = pose.t.reshape(1,3)

        # find positives and negatives 
        def metric(v1, v2):
            dxy = math.sqrt((v1[4]-v2[4])**2 + (v1[5]-v2[5])**2)
            dz = abs(v1[6] - v2[6])

            q1 = quaternion.as_quat_array(v1[0:4])
            q2 = quaternion.as_quat_array(v2[0:4])

            R1 = quaternion.as_rotation_matrix(q1)
            R2 = quaternion.as_rotation_matrix(q2)
            dR = R2.T @ R1

            dtheta = abs(np.arccos((np.matrix.trace(dR)-1)/2))
            
            if dz > 1.5:
                return 100
            elif dtheta > np.pi / 4:
                return 100
            
            return dxy

        kNN = NearestNeighbors(algorithm='brute', metric=metric)
        kNN.fit(self.utmDB)
        D, I = kNN.radius_neighbors(self.utmQ, pos_threshold)
        nD, nI = kNN.radius_neighbors(self.utmQ, neg_threshold)
        
        qIdx = []
        pIdx = []
        nonNegIdx = []
        for i in range(D.shape[0]):
            if D[i].shape[0] == 0:
                continue
            qIdx.append(i)
            pIdx.append(I[i][D[i].argsort()])    
            nonNegIdx.append(nI[i][nD[i].argsort()])


        self.qIdx = np.asarray(qIdx)
        self.pIdx = np.asarray(pIdx, dtype=object)
        self.nonNegIdx = np.asarray(nonNegIdx, dtype=object)

        nIdx = []
        for pos in nonNegIdx:
            nIdx.append(np.setdiff1d(np.arange(len(db_image_names)), pos, assume_unique=True))
        self.nIdx = np.asarray(nIdx, dtype=object)

        self.qImages = np.asarray([name for _,name in query_image_names])
        self.dbImages = np.asarray([name for _,name in db_image_names])

    def update_subcache(self, net = None):
        self.triplets = []
        if net is None:
            for q in range(len(self.qIdx)):
                qidx = self.qIdx[q]
                pidx = np.random.choice(self.pIdx[q], size=1)[0]

                while True:
                    nidxs = np.random.choice(len(self.dbImages), size=5)
                    if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                        break                
                triplet = [qidx, pidx, *nidxs]
                self.triplets.append(triplet)
        else:
            print('===> hard triplets mining')
            opt = {'batch_size': 128, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
            qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages, transform=self.transform),**opt)
            dbloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages, transform=self.transform),**opt)

            # calculate their descriptors
            net.eval()
            with torch.no_grad():
                # initialize descriptors
                qvecs = torch.zeros(len(self.qImages), 32768).to('cuda')
                dbvecs = torch.zeros(len(self.dbImages), 32768).to('cuda')

                bs = 128

                # compute descriptors
                for iteration, (input, indices) in tqdm(enumerate(qloader), desc='compute query descriptors'):
                    input = input.to('cuda')
                    image_encoding = net.encoder(input)
                    vlad_encoding = net.pool(image_encoding) 

                    qvecs[indices, :] = vlad_encoding
                    # qvecs[i*bs:(i+1)*bs, : ] = net(X.to(self.device)).data

                for iteration, (input, indices) in tqdm(enumerate(dbloader), desc='compute db descriptors'):
                    input = input.to('cuda')
                    image_encoding = net.encoder(input)
                    vlad_encoding = net.pool(image_encoding) 

                    dbvecs[indices, :] = vlad_encoding
                   
                print(f'qvec: {qvecs.shape}, dbvec: {dbvecs.shape}')
            
            print('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            Scores = torch.mm(qvecs, dbvecs.t())
            Scores, Ranks = torch.sort(Scores, dim=1, descending=True)

            # convert to cpu and numpy
            Scores, Ranks = Scores.cpu().numpy(), Ranks.cpu().numpy()
            print(f'Scores: {Scores.shape}, Ranks: {Ranks.shape}')
            
            self.Scores = Scores
            self.Ranks = Ranks

            # selection of hard triplets
            for q in range(len(self.qIdx)):

                qidx = self.qIdx[q]

                # find positive idx for this query (cache idx domain)
                cached_pidx = self.pIdx[qidx]
                cached_nidx = self.nIdx[qidx]

                # find idx of positive idx in rank matrix (descending cache idx domain)
                pidx = np.where(np.in1d(Ranks[q,:], cached_pidx))[0]
                nidx = np.where(np.in1d(Ranks[q,:], cached_nidx))[0]

                # take the closest positve
                dPos = Scores[q, pidx][0]

                # get distances to all negatives
                dNeg = Scores[q, nidx]
                NegRanks = Ranks[q, nidx]

                # how much are they violating
                loss = dPos - dNeg + self.margin ** 0.5
                violatingNeg = 0 < loss

                # if less than nNeg are violating then skip this query
                if np.sum(violatingNeg) <= self.nNeg: continue

                # select hardest negatives
                hardest_negIdx = np.argsort(loss)[0:self.nNeg]

                # select the hardest negatives
                cached_hardestNeg = NegRanks[hardest_negIdx]

                # select the closest positive (back to cache idx domain)
                cached_pidx = Ranks[q, pidx][0]

                # transform back to original index (back to original idx domain)
                qidx = qidx
                pidx = cached_pidx
                hardestNeg = cached_hardestNeg

                # package the triplet and target
                triplet = [qidx, pidx, *hardestNeg]

                self.triplets.append(triplet)
        return

    def __getitem__(self, index):
        triplet = self.triplets[index]
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # load images into triplet list
        output = [torch.stack([self.transform(Image.open(im)) for im in self.qImages[qidx].split(',')])]
        output.append(torch.stack([self.transform(Image.open(im)) for im in self.dbImages[pidx].split(',')]))
        output.extend([torch.stack([self.transform(Image.open(im)) for im in self.dbImages[idx].split(',')]) for idx in nidx])        
    
        return torch.cat(output)

    def __len__(self):
        return len(self.triplets)
    