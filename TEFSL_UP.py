import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
from models import *
from torch.nn.parameter import Parameter
# import spectral
import modelStatsRecord
import ScConv
# from renet import RENet
from einops import rearrange
from cca import CCA
from operator import itemgetter
from tqdm.notebook import tqdm
use_gpu = torch.cuda.is_available()
# from torchsummary import summary
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 128)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 103) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 9)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
# target
parser.add_argument("-m","--test_class_num",type=int, default=9)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')
# test
parser.add_argument("-t","--test_queries_num_per_class",type=int,default=5)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1
TEST_QUERIES_NUM_PER_CLASS=args.test_queries_num_per_class

#protolp parameters
n_shot = TEST_LSAMPLE_NUM_PER_CLASS
n_ways = TEST_CLASS_NUM
n_queries = TEST_QUERIES_NUM_PER_CLASS
n_runs=1#constant
n_lsamples = n_ways * n_shot
n_usamples = n_ways * n_queries
n_samples = n_lsamples + n_usamples
def centerDatas(datas):
    m=datas.mean(1, keepdim=True)
    datas= datas - m
    nor=torch.norm(datas, dim=2, keepdim= True)
    datas = datas / torch.norm(datas, dim=2, keepdim= True)

    return datas,m,nor

def scaleEachUnitaryDatas(datas):
  
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms

def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1),'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas

def SVDreduction(ndatas,K):
    # ndatas = torch.linear.qr(datas.permute(0, 2, 1),'reduced').R
    # ndatas = ndatas.permute(0, 2, 1)
    _,s,v = torch.svd(ndatas)
    ndatas = ndatas.matmul(v[:,:,:K])

    return ndatas


def predict(gamma, Z, labels):
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    delta = torch.sum(Z, 1)
    iden = torch.eye(n_ways,device='cuda')
    iden = iden.reshape((1, n_ways, n_ways))###########task=10000
    iden = iden.repeat(1, 1, 1)
    W = torch.bmm(torch.transpose(Z,1,2), Z/delta.unsqueeze(1))
    L = iden - W
    Z_l = Z[:,:n_lsamples]
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L + 0.2*iden)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    Pred = Z.bmm(A)
    normalizer = torch.sum(Pred,dim=1,keepdim=True)
    Pred = (n_shot+n_queries)*Pred/normalizer

    return Pred#.clamp(0,1)

def predictW(gamma, Z, labels):
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    delta = torch.sum(Z, 1)
    iden = torch.eye(n_ways,device='cuda')
    iden = iden.reshape((1, n_ways, n_ways))
    iden = iden.repeat(1, 1, 1)
    W = torch.bmm(torch.transpose(Z,1,2), Z/delta.unsqueeze(1))
    L = iden - (W + W.bmm(W))/2
    Z_l = Z[:,:n_lsamples]
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L + 1*iden)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    P = Z.bmm(A)
    _, n, m = P.shape
    r = torch.ones(n_runs, n_lsamples + n_usamples,device='cuda')
    c = torch.ones(n_runs, n_ways,device='cuda') * (n_shot + n_queries)
    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
    while torch.max(torch.abs(u - P.sum(2))) > 0.01:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        P[:,:n_lsamples].fill_(0)
        P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        if iters == maxiters:
            break
        iters = iters + 1
    return P

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self, ndatas, n_runs, n_shot, n_queries, n_ways, n_nfeat):
        self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:n_shot,].mean(1)
        self.mus = self.mus_ori.clone()

    def updateFromEstimate(self, estimate, alpha, l2 = False):

        diff = self.mus_ori - self.mus
        Dmus = estimate - self.mus
        if l2 == True:
            self.mus = self.mus + alpha * (Dmus) + 0.01 * diff
        else:
            self.mus = self.mus + alpha * (Dmus)
        return self.mus

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        #print(P.shape,P.view((n_runs, -1)).shape,P.view((n_runs, -1)).sum(1).unsqueeze(1).shape,P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1).shape)
        P =P/P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
                                         
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P = P* (r / u).view((n_runs, -1, 1))
            P =P* (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, ndatas, n_runs, n_ways, n_usamples, n_lsamples):
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * (n_queries)
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-3)
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        p_xj=p_xj/torch.sum(p_xj, dim=2, keepdim=True)
        return p_xj

    def estimateFromMask(self, mask, ndatas):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, epochInfo=None):
     
        p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas,ndatas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)
        if self.verbose:
            op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=20):
        
        self.probas = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)#Z
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            self.probas = p_xj
            pesudo_L = predictW(1, self.probas, labels)
            beta = 0.6
            m_estimates = model.estimateFromMask((beta*pesudo_L + (1-beta)*self.probas).clamp(0,1), ndatas)
            # update centroids
            cnter=model.updateFromEstimate(m_estimates, self.alpha)
            if self.verbose:
                op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
                acc = self.getAccuracy(op_xj)
            if (self.progressBar): pb.update()
        # get final accuracy and return it
        op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        acc = self.getAccuracy(op_xj)
        return acc,op_xj,cnter

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
source_imdb['data']=np.array(source_imdb['data'])
source_imdb['Labels']=np.array(source_imdb['Labels'],dtype='int')
source_imdb['set']=np.array(source_imdb['set'],dtype='int')
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']# (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(np.array(source_imdb['data']).shape) # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
last_dir = os.path.abspath(os.path.dirname(os.getcwd()))
test_data = last_dir+'/HSI_META_DATA/test/paviaU.mat'
test_label = last_dir+'/HSI_META_DATA/test/paviaU_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = next(train_loader.__iter__())
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain

# model
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
def conv3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class Octave_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(Octave_3d_CNN, self).__init__()

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channel, out_channel1, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(out_channel1),
            nn.ReLU(),
        )
        
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*98, out_channels=out_channel2, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU(),
        )
        
        self.scconv=ScConv.ScCon(out_channel2)
        
        self.final_feat_dim = FEATURE_DIM

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x1 = self.conv2d_features(x)
        x2 = self.scconv(x1)+x1
        # flatten = F.adaptive_avg_pool2d(x2,(1,1))

        # embedding_feature = flatten.view(flatten.shape[0],-1) #(1,160)

        return x2
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)
    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.feature_encoder = Octave_3d_CNN(1,8,128)
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x
            
    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)    
               
    def cca(self, spt, qry):
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # (S * C * Hs * Ws, Q * C * Hq * Wq) -> Q * S * Hs * Ws * Hq * Wq
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        # corr4d refinement
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / 5, dim=2)   ####args.temperature_attn
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / 5, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[4, 5])#[171,9,5,5]
        attn_q = corr4d_q.sum(dim=[2, 3])#[171,9,5,5]

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0) #[171,9,1,5,5]*[1,9,128,5,5]=[171,9,128,5,5]
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1) #[171,9,1,5,5]*[1,9,128,5,5]=[171,9,128,5,5]

        # averaging embeddings for k > 1 shots
        if args.shot_num_per_class > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2]) #[171,9,128]
        spt_attended_pooled = spt_attended_pooled.mean(dim=[0])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2]) #[171,9,128]
        qry_attended_pooled = qry_attended_pooled.mean(dim=[1])
        return spt_attended_pooled, qry_attended_pooled       
           
    def get_4d_correlation_map(self, spt, qry):
        '''
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        '''
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum
    
    def forward(self, spt, qry, domain='source'):  # x
        # print(x.shape)
        if domain == 'target':
            spt = self.target_mapping(spt)  
            qry = self.target_mapping(qry)
        elif domain == 'source':
            spt = self.source_mapping(spt) 
            qry = self.source_mapping(qry)
        # print(x.shape)#torch.Size([45, 100, 9, 9])
        spt_mid = self.feature_encoder(spt)  # (9,5,5,128)
        qry_mid = self.feature_encoder(qry)  # (9*19,5,5,128)
        spt_attended, qry_attended = self.cca(spt_mid, qry_mid)  #MFIM 
        # print((feature.shape))
        return spt_attended, qry_attended 
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
latest_G,latest_RandPerm,latest_Row, latest_Column,latest_nTrain = None,None,None,None,None
seeds = [1337, 1220, 1336, 1330, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    print(get_parameter_number(feature_encoder))

    feature_encoder.apply(weights_init)

    feature_encoder.cuda()
    # summary(feature_encoder, (100, 9, 9))
    feature_encoder.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(EPISODE):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = next(source_iter)
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = next(source_iter)

        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)

        # source domain few-shot
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5, 1, 15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = next(support_dataloader.__iter__())  # (5, 100, 9, 9)
            querys, query_labels = next(query_dataloader.__iter__())  # (75,100,9,9)

            # calculate features
            support_features, query_features = feature_encoder(supports.cuda(), querys.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())
            loss = f_loss

            # Update parameters
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = next(support_dataloader.__iter__())  # (5, 100, 9, 9)
            querys, query_labels = next(query_dataloader.__iter__())  # (75,100,9,9)

            # calculate features
            support_features, query_features = feature_encoder(supports.cuda(), querys.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            loss = f_loss
            
            # Update parameters
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            elapsed_time = time.time()-train_start
            train_loss.append(loss.item())
            print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}, elapsed time: {:6.4f}'.format(episode + 1, \
                                                                                                                loss.item(),
                                                                                                                querys.shape[0],
                                                                                                                total_hit / total_num,
                                                                                                                   elapsed_time))
        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            labels_al = np.array([], dtype=np.int64)
            
            train_datas, train_labels = next(train_loader.__iter__())
            test_datas, _ = next(test_loader.__iter__())
            train_features, _ = feature_encoder(Variable(train_datas).cuda(), Variable(test_datas).cuda(), domain='target')  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            support_proto=train_features.reshape(-1,TEST_LSAMPLE_NUM_PER_CLASS,FEATURE_DIM).permute(1,0,2)
            support_proto=support_proto.mean(0)
            support_proto=support_proto/support_proto.norm(dim=1, keepdim=True)
            
            logits_pseudo=np.zeros((1,TEST_CLASS_NUM))##(1,9)
            features_pseudo=np.zeros((1,FEATURE_DIM))##(1,128)
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                _, test_features = feature_encoder(Variable(train_datas).cuda(), Variable(test_datas).cuda(), domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                logits = euclidean_metric(test_features, support_proto)
                #logits=softmax(logits)##softmax
                
                predict_labels = torch.argmax(logits.detach(), dim=1).cpu()
                test_labels = test_labels.numpy()
                labels = np.append(labels, predict_labels)
                
                labels_al = np.append(labels_al, test_labels)
                logits_pseudo=np.append(logits_pseudo,logits.detach().cpu(),axis=0)
                features_pseudo=np.append(features_pseudo,test_features.detach().cpu(),axis=0)
                counter += batch_size
                
            logits_pseudo=logits_pseudo[1:,:]
            features_pseudo=features_pseudo[1:,:]
            print("size of test features",features_pseudo.shape)
            print("size of logits ",logits_pseudo.shape)
            print("size of labels",labels.shape)
            
            print(np.amax(logits_pseudo,axis=1).shape)
            pseudo=np.concatenate((labels.reshape(-1,1),np.amax(logits_pseudo,axis=1).reshape(-1,1),features_pseudo),axis=1)
            print(pseudo.shape)#(42731,1+1+128)
            print(train_labels,train_features.shape)
            pseudo_train_features=train_features.reshape(TEST_CLASS_NUM,TEST_LSAMPLE_NUM_PER_CLASS,int(train_features.shape[1]))#(9,5,128)
            pseudo_ways_all=torch.tensor(np.zeros((1,n_queries,FEATURE_DIM)))    
            for i in range(TEST_CLASS_NUM):
                pos=np.where(pseudo[:,0]==i)
                pseudo_ways=pseudo[pos,1:].reshape(-1,1+FEATURE_DIM)
                #print(pseudo_ways[:2,:])
                pseudo_ways=torch.tensor(sorted(pseudo_ways,key=itemgetter(0),reverse=True))
                pseudo_ways=pseudo_ways[:n_queries,1:].reshape(1,n_queries,-1)
                pseudo_ways_all=torch.cat((pseudo_ways_all,pseudo_ways),dim=0)
                
            pseudo_train_features=torch.cat((pseudo_train_features.cuda(),pseudo_ways_all[1:,:,:].cuda()),dim=1).permute(1,0,2)#(15,9,160)
            print(pseudo_train_features.shape)    
            pseudo_train_features=pseudo_train_features.reshape(1,-1,FEATURE_DIM)
            z=torch.zeros(n_shot+n_queries,TEST_CLASS_NUM)
            pseudo_train_labels=torch.tensor(range(0,TEST_CLASS_NUM)).reshape(1,-1).expand_as(z).reshape(1,-1)
            pseudo_train_features=torch.tensor(pseudo_train_features,dtype=torch.float32)
            pseudo_train_labels=torch.tensor(pseudo_train_labels,dtype=torch.int64)
            print("data needed")
            print(pseudo_train_features.shape,pseudo_train_labels.shape,features_pseudo.shape)
            features_pseudo=torch.tensor(features_pseudo,dtype=torch.float32).cuda()
            print("****************************************")  
            ndatas=torch.cat((pseudo_train_features,features_pseudo.reshape(1,-1,FEATURE_DIM)),dim=1)
            test_features=ndatas[:,(n_shot+n_queries)*TEST_CLASS_NUM:,:]
            ndatas=ndatas[:,:(n_shot+n_queries)*TEST_CLASS_NUM,:]
            n_nfeat = ndatas.size(2)
            labels=pseudo_train_labels
            
            # switch to cuda
            ndatas = ndatas.cuda()
            labels = labels.cuda()
    
            #MAP
            lam = 10
            model = GaussianModel(n_ways, lam)
            model.initFromLabelledDatas(ndatas, n_runs, n_shot,n_queries,n_ways,n_nfeat)#initial Ck
    
            alpha = 0.2
            optim = MAP(alpha)

            optim.verbose=False
            optim.progressBar=False
            T1 = time.perf_counter()
            
            acc_test,op_xj,mus= optim.loop(model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=9)
            test_features=test_features.reshape(-1,FEATURE_DIM)
            mus=mus.reshape(-1,FEATURE_DIM)
            dist = euclidean_metric(test_features, mus)
            dist_train=euclidean_metric(train_features.reshape(-1,FEATURE_DIM), mus)
            predict_labels = torch.argmax(dist, dim=1).cpu()
            print("shape is",predict_labels.shape)
            
            rewards = [1 if predict_labels[j] == labels_al[j] else 0 for j in range(counter)]

            total_rewards = np.sum(rewards)

            predict = predict_labels
            accuracy = total_rewards / 1.0 / counter  #
            accuracies.append(accuracy)    
            print(accuracy)
            ###########################################################################################
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str( "checkpoints/TEFSL_feature_encoder_" + "UP_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = total_rewards / len(test_loader.dataset)
                OA = acc[iDataSet]
                C = metrics.confusion_matrix(labels_al, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels_al, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
    training_time[iDataSet] = train_end - train_start
    test_time[iDataSet] = test_end - train_end

    latest_G, latest_RandPerm, latest_Row, latest_Column, latest_nTrain = G, RandPerm, Row, Column, nTrain
    for i in range(len(predict)):  # predict ndarray <class 'tuple'>: (9729,)
        latest_G[latest_Row[latest_RandPerm[latest_nTrain + i]]][latest_Column[latest_RandPerm[latest_nTrain + i]]] = \
            predict[i] + 1
    sio.savemat('classificationMap/UP/TEFSL_UP_pred_map_latest' + '_' + repr(int(OA * 10000)) + '.mat', {'latest_G': latest_G})
    hsi_pic_latest = np.zeros((latest_G.shape[0], latest_G.shape[1], 3))
    for i in range(latest_G.shape[0]):
        for j in range(latest_G.shape[1]):
            if latest_G[i][j] == 0:
                hsi_pic_latest[i, j, :] = [0, 0, 0]
            if latest_G[i][j] == 1:
                hsi_pic_latest[i, j, :] = [216, 191, 216]
            if latest_G[i][j] == 2:
                hsi_pic_latest[i, j, :] = [0, 255, 0]
            if latest_G[i][j] == 3:
                hsi_pic_latest[i, j, :] = [0, 255, 255]
            if latest_G[i][j] == 4:
                hsi_pic_latest[i, j, :] = [45, 138, 86]
            if latest_G[i][j] == 5:
                hsi_pic_latest[i, j, :] = [255, 0, 255]
            if latest_G[i][j] == 6:
                hsi_pic_latest[i, j, :] = [255, 165, 0]
            if latest_G[i][j] == 7:
                hsi_pic_latest[i, j, :] = [159, 31, 239]
            if latest_G[i][j] == 8:
                hsi_pic_latest[i, j, :] = [255, 0, 0]
            if latest_G[i][j] == 9:
                hsi_pic_latest[i, j, :] = [255, 255, 0]
    utils.classification_map(hsi_pic_latest[4:-4, 4:-4, :] / 255, latest_G[4:-4, 4:-4], 24,
                             'classificationMap/UP/TEFSL_UP_pred_map_latest'+ '_' + repr(int(OA * 10000))+'.png')

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
###
ELEMENT_ACC_RES_SS4 = np.transpose(A)
AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4,0)
OA_RES_SS4 = np.transpose(acc)
KAPPA_RES_SS4 = np.transpose(k)
ELEMENT_PRE_RES_SS4 = np.transpose(P)
AP_RES_SS4= np.mean(ELEMENT_PRE_RES_SS4,0)
TRAINING_TIME_RES_SS4 = np.transpose(training_time)
TESTING_TIME_RES_SS4 = np.transpose(test_time)
classes_num = TEST_CLASS_NUM
ITER = nDataSet

modelStatsRecord.outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                              ELEMENT_PRE_RES_SS4, AP_RES_SS4,
                              TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                              classes_num, ITER,
                              './records/pavia_result_train_iter_times_{}shot_Chikusei_iter_9times_0.2.txt'.format(TEST_LSAMPLE_NUM_PER_CLASS))##centernot
