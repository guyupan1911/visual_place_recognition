import sys
sys.path.append('../thirdParty/mapillary_sls')

from pathlib import Path
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.datasets.generic_dataset import ImagesFromList
from mapillary_sls.utils.utils import configure_transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mapillary_sls.utils.visualize import denormalize, visualize_triplets
from mapillary_sls.utils.eval import download_msls_sample

root_dir = Path('/autox-dl/localization/guyu/dataset/MSLS/').absolute()

if not root_dir.exists():
    download_msls_sample(root_dir)

# get transform
meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
transform = configure_transform(image_dim = (480, 640), meta = meta)


def get_train_set(SAMPLE_CITIES, cacheRefreshRate):
    # positive are defined within a radius of 5 m
    posDistThr = 5

    # negatives are defined outside a radius of 25 m
    negDistThr = 25

    # number of negatives per triplet
    nNeg = 1

    # number of cached queries
    cached_queries = cacheRefreshRate

    # number of cached negatives
    cached_negatives = 100

    # whether to use positive sampling
    positive_sampling = True

    # choose the cities to load
    cities = SAMPLE_CITIES

    # choose task to test on [im2im, seq2im, im2seq, seq2seq]
    task = 'im2im'

    # choose sequence length
    seq_length = 1

    # choose subtask to test on [all, s2w, w2s, o2n, n2o, d2n, n2d]
    subtask = 'all'

    dataset = MSLS(root_dir, cities = cities, transform = transform, mode = 'train', task = task, seq_length = seq_length,
                    negDistThr = negDistThr, posDistThr = posDistThr, nNeg = nNeg, cached_queries = cached_queries,
                    cached_negatives = cached_negatives, positive_sampling = positive_sampling)
    
    return dataset


def get_test_set(SAMPLE_CITIES=''):
    # positive are defined within a radius of 25 m
    posDistThr = 25

    # choose task to test on [im2im, seq2im, im2seq, seq2seq]
    task = 'im2im'

    # choose sequence length
    seq_length = 1

    # choose subtask to test on [all, s2w, w2s, o2n, n2o, d2n, n2d]
    subtask = 'all'

    dataset = MSLS(root_dir, cities = SAMPLE_CITIES, transform = transform, mode = 'val',
                      task = task, seq_length = seq_length, subtask = subtask, posDistThr = posDistThr)
    return ImagesFromList(dataset.qImages[dataset.qIdx], transform), \
           ImagesFromList(dataset.dbImages, transform), dataset.pIdx 
