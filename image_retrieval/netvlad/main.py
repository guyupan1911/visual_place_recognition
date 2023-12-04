import argparse
import sys
sys.path.append('/workspace/visual_place_recognition/datasets')
import netvlad
import autox as dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from tensorboardX import SummaryWriter

import random, shutil, json
from math import log10, ceil
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
from datetime import datetime
import numpy as np
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--encoder_dim', type=int, default=512, help='encoder demision Defaut=64')
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--dataPath', type=str, default='data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--evalEvery', type=int, default=5, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--batchSize', type=int, default=8, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')


def train(epoch):
    epoch_loss = 0
    nBatches = ceil(len(train_set) / opt.batchSize)
    # print(f'nBatches {nBatches}')  

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True)

    model.train()
    for iteration, sequences in enumerate(tqdm(training_data_loader)):
        # print(f'iteration {iteration}')
        B, N, C, H, W = sequences.shape
        
        input = sequences.reshape(-1,C,H,W)

        input = input.to('cuda')
        image_encoding = model.encoder(input)
        vlad_encoding = model.pool(image_encoding) 

        vlad = torch.split(vlad_encoding, N)

        optimizer.zero_grad()
        
        # calculate loss for each Query, Positive, Negative triplet
        # due to potential difference in number of negatives have to 
        # do it per query, per negative
        loss = 0
        # n_positive = 0
        # n_negative = 0
        for i in range(len(vlad)):
            for j in range(5):
                loss += criterion(vlad[i][0], vlad[i][1], vlad[i][2+j])
            # if (l > 0):
            #     n_positive += 1
            # else:
            #     n_negative += 1
        

        loss  /= torch.tensor(5*B).float().to('cuda') # normalise by actual number of negatives
        # print(f'n_positive {n_positive}, n_negative {n_negative}, loss {loss}')
        loss.backward()
        optimizer.step()
        del input, image_encoding, vlad_encoding, vlad

        batch_loss = loss.item()
        epoch_loss += batch_loss

        if iteration % 50 == 0 or nBatches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                nBatches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * nBatches) + iteration)
            # writer.add_scalar('Train/nNeg', nNeg, 
            #         ((epoch-1) * nBatches) + iteration)

    del training_data_loader, loss
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test(q_set, db_set, pIdx, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory? 
    query_data_loader = DataLoader(dataset=q_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=True)

    database_data_loader = DataLoader(dataset=db_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=True)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = opt.encoder_dim * opt.num_clusters
        qFeat = np.empty((len(q_set), pool_size))
        dbFeat = np.empty((len(db_set), pool_size))

        for iteration, (input, indices) in enumerate(query_data_loader, 0):
            input = input.to('cuda')
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            qFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(query_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(query_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
        
        for iteration, (input, indices) in enumerate(database_data_loader, 0):
            input = input.to('cuda')
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(database_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(database_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding

    del query_data_loader, database_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = qFeat.astype('float32')
    dbFeat = dbFeat.astype('float32')
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    # n_values = [1,5,10,20]
    n_values = [1]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 

    # for each query get those within threshold distance
    gt = pIdx

    correct_at_n = np.zeros(len(n_values))
    
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
            # if no success candidates
            print(f'qIdx: {qIx}, pred:{pred[0]}, gt: {gt[qIx]}')
            fig = plt.figure(figsize=(18,5))
            qImage = Image.open(q_set.images[qIx]).resize((640, 640))
            plt.subplot(1,3,1)
            plt.imshow(qImage)
            dbImage = Image.open(db_set.images[pred[0]]).resize((640,640))
            plt.subplot(1,3,2)
            plt.imshow(dbImage)
            pImage = Image.open(db_set.images[gt[qIx][0]]).resize((640,640))
            plt.subplot(1,3,3)
            plt.imshow(pImage)
            plt.savefig(f'data/results/{qIx}.png')
            plt.close()

    recall_at_n = correct_at_n / len(q_set)

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

if __name__ == "__main__":
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = netvlad.create_model(mode=opt.mode, pretrained=not opt.fromscratch, resume=opt.resume)

    print('===> Loading dataset(s)')
    if opt.mode == 'test':
        q_set, db_set, pIdx = dataset.get_test_set()
        print(f'test mode q_set size: {len(q_set)}, db_set: {len(db_set)}')
    elif opt.mode == 'train':
        q_set, db_set, pIdx = dataset.get_test_set()
        print(f'train mode val q_set size: {len(q_set)}, db_set: {len(db_set)}')
        train_set = dataset.get_train_set()

    if opt.mode == 'test':
        print('===> Running evaluation step')
        recalls = test(q_set, db_set, pIdx, write_tboard=False)
    elif opt.mode == 'train':
        print('===> Training model')
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to('cuda')

        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        best_score = 0
        for epoch in range(1, opt.nEpochs+1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train_set.update_subcache(model)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(q_set, db_set, pIdx, epoch, write_tboard=True)
                is_best = recalls[1] > best_score 
                if is_best:
                    best_score = recalls[1]
                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                }, is_best)
        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()

