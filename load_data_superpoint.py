import numpy as np
import torch
import os
import cv2
import math
import datetime
import time
import random
import torch.nn as nn

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from models.superpoint import SuperPoint

class SparseDataset_SP(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures):

        self.device = 'cuda:7'
        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]

        self.nfeatures = nfeatures
        self.superpoint = SuperPoint({'max_keypoints':nfeatures}).eval().to(self.device)

        #self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

	# load precalculated correspondences
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
        image = image.astype('float32')

        width, height = image.shape[:2]
        # max_size = max(width, height)
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)

        match_size = 0
        while match_size < 5:
            warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

            M = cv2.getPerspectiveTransform(corners, corners + warp)
            warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0])) # return an image type
        
            #sp_image = image.astype('float32')
            sp_image = torch.from_numpy(image / 255.).float()[None, None].to(self.device)
            result_image = self.superpoint({'image': sp_image})

            #warp_image = warped.astype('float32')
            warp_image = torch.from_numpy(warped / 255.).float()[None, None].to(self.device)
            result_warped = self.superpoint({'image': warp_image})

            #print(result_warped['keypoints'][0].shape)
            #print(result_warped['descriptors'][0].shape)

            len_kp1 = result_image['keypoints'][0].shape[0]
            len_kp2 = result_warped['keypoints'][0].shape[0]
            kp1_num = min(self.nfeatures, len_kp1)
            kp2_num = min(self.nfeatures, len_kp2)
            #print('kpt numbers : ', kp1_num, kp2_num)

            #convert tensor keypoints to numpy keypoints
            kp1_np = result_image['keypoints'][0].cpu().numpy() # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
            kp2_np = result_warped['keypoints'][0].cpu().numpy()
            #print('kp2 shape', kp2_np.shape)

            if len_kp1 < 1 or len_kp2 < 1:
                continue

            #print('score : ', result_image['scores'])
            scores1_np = result_image['scores'][0].cpu().detach().numpy().copy() # confidence of each key point
            scores2_np = result_warped['scores'][0].cpu().detach().numpy().copy()
            #print('score : ', scores1_np.shape)

            kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] # why [0, :, :]
            dists = cdist(kp1_projected, kp2_np)

            min1 = np.argmin(dists, axis=0)
            min2 = np.argmin(dists, axis=1)

            descs1 = result_image['descriptors'][0].cpu().detach().numpy()
            descs2 = result_warped['descriptors'][0].cpu().detach().numpy()

            #random added keypoints for batches
            if kp1_num < self.nfeatures:
                rand_x = np.random.randint(0, image.shape[1], size=(self.nfeatures - kp1_num, 1)).astype(np.float32)
                rand_y = np.random.randint(0, image.shape[0], size=(self.nfeatures - kp1_num, 1)).astype(np.float32)
                rand = np.concatenate((rand_x, rand_y), axis=1)
                kp1_np = np.concatenate((kp1_np, rand), axis=0)
                #descs
                desc = np.random.randint(0, 10000, size=(256, self.nfeatures - kp1_num)).astype(np.float32)
                row_sums = desc.sum(axis=1)
                if not row_sums[:, np.newaxis].any() < 0.001:
                    desc = desc / row_sums[:, np.newaxis]
                else:
                    print('desc is : ', row_sums[:, np.newaxis])
                descs1 = np.concatenate((descs1, desc), axis=1)
                #scores
                scores1_np.resize(self.nfeatures)
                kp1_num = kp1_np.shape[0]

            if kp2_num < self.nfeatures:
                rand_x = np.random.randint(0, warped.shape[1], size=(self.nfeatures - kp2_num, 1)).astype(np.float32)
                rand_y = np.random.randint(0, warped.shape[0], size=(self.nfeatures - kp2_num, 1)).astype(np.float32)
                rand = np.concatenate((rand_x, rand_y), axis=1)
                kp2_np = np.concatenate((kp2_np, rand), axis=0)
                #descs
                desc = np.random.randint(0, 10000, size=(256, self.nfeatures - kp2_num)).astype(np.float32)
                row_sums = desc.sum(axis=1)
                if not row_sums[:, np.newaxis].any() < 0.001:
                    desc = desc / row_sums[:, np.newaxis]
                else:
                    print('desc is : ', row_sums[:, np.newaxis])
                descs2 = np.concatenate((descs2, desc), axis=1)
                #scores
                scores2_np.resize(self.nfeatures)
                kp2_num = kp2_np.shape[0]


            start = time.time()
            matches_debug = np.array([[i, min2[i]] for i in range(len(min2)) if (min1[min2[i]] == i and dists[i, min2[i]] < 3)])
            if matches_debug.shape[0] < 5:
                print("no matches: ",file_name)
                continue

            matches_miss1 = np.array([[i, kp2_num] for i in range(kp1_num) if i not in matches_debug[:, 0]])
            matches_miss2 = np.array([[kp1_num, i] for i in range(kp2_num) if i not in matches_debug[:, 1]])
            all_matches_ = np.concatenate([matches_debug, matches_miss1, matches_miss2], axis=0)
            all_matches_ = all_matches_[:1500]
            if all_matches_.shape[0] < 1500:
                add_zeros = np.full((1500 - all_matches_.shape[0], 2), kp1_num, dtype=np.int32)
                #print('before zeros to matches : ', all_matches_.shape, add_zeros.shape)
                all_matches_ = np.concatenate([all_matches_, add_zeros], axis=0)
                #print('after zeros to matches : ', all_matches_.shape)
            all_matches = np.transpose(all_matches_)
            
            visualize = False 
            if visualize:

                #draw matches
                img_all = (np.hstack((image,warped))).astype('uint8')
                img_all = cv2.cvtColor(img_all,cv2.COLOR_GRAY2BGR)
                for match in matches_debug:
                    #img_debug = img_all.copy()
                    pt1 = kp1_np[int(match[0]), :]
                    pt2 = kp2_np[int(match[1]), :]
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    #print(p1)
                    p2 = (int(round(pt2[0] + image.shape[1])), int(round(pt2[1])))
                    cv2.line(img_all, p1, p2, (0,255,0), thickness=1, lineType=1)
                cv2.imshow('image', img_all)
                cv2.waitKey()


            kp1_np = torch.from_numpy(kp1_np.reshape((-1, 2)))
            kp2_np = torch.from_numpy(kp2_np.reshape((-1, 2)))
            descs1 = torch.from_numpy(descs1.reshape((descs1.shape[0], descs1.shape[1])))
            descs2 = torch.from_numpy(descs2.reshape((descs2.shape[0], descs2.shape[1])))
            scores1_np = torch.from_numpy(scores1_np)
            scores2_np = torch.from_numpy(scores2_np)
            all_matches = torch.from_numpy(all_matches)
            #file_name = 

            # descs1 = np.transpose(descs1 / 256.)
            # descs2 = np.transpose(descs2 / 256.)
            #descs1 = result_image['descriptors'][0].detach().numpy()
            #descs2 = result_warped['descriptors'][0].detach().numpy()
            

            image = torch.from_numpy(image/255.).double()[None].cuda()
            warped = torch.from_numpy(warped/255.).double()[None].cuda()

            #scores1_np = scores1_np.reshape((1,-1,1))
            #scores2_np = scores2_np.reshape((1,-1,1))

            '''
            descs1 = np.transpose(descs1)
            descs1 = descs1.reshape((1,descs1.shape[0],descs1.shape[1]))
            descs2 = np.transpose(descs2)
            descs2 = descs2.reshape((1,descs2.shape[0],descs2.shape[1]))
            '''

            '''
            print('kp1_np shape : ', kp1_np.shape)
            print('kp1_np  : ', type(kp1_np))
            print('descs1 shape : ', descs1.shape)
            print('descs1 : ', type(descs1))
            print('scores0 shape : ', scores1_np.shape)
            print('scores0 : ', type(scores1_np))
            print('all_matches.shape ' , all_matches.shape)
            print('all_matches ' , type(all_matches))
            print('image 0 ', image.shape)
            print('image 1 ', warped.shape)
            '''

            return{
                'keypoints0': kp1_np,
                'keypoints1': kp2_np,
                'descriptors0': (descs1),
                'descriptors1': (descs2),
                'scores0': (scores1_np),
                'scores1': (scores2_np),
                'image0': image,
                'image1': warped,
                'all_matches': (all_matches),
                'file_name': file_name
            }
    

'''
train_set = SparseDataset_SP('./train_data/', 1024)
train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=1, drop_last=True)

for i, pred in enumerate(train_loader):
    #print(pred)
    print('done!')
'''

'''
kp1_np shape :  (1, 904, 2)
descs1 shape :  (128, 904)
scores0 shape :  (904,)
'''
