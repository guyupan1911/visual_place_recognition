import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import kapture_localization.utils.logging
from kapture_localization.utils.pairsfile import get_pairs_from_file

import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_from_file, get_all_tar_handlers
from kapture.io.records import get_image_fullpath

from utils.pairsfile import readImagePairs

logger = logging.getLogger('visualize image pairs')

def statistics_analysis(image_pairs):
    N = len(image_pairs)
    min_distances = np.zeros((N))
    max_distances = np.zeros((N))
    median_distances = np.zeros((N))
    mean_distances = np.zeros((N))
    top1_distances = np.zeros((N))

    for i, (query, matches) in enumerate(image_pairs.items()):
        distances = np.zeros((len(matches)))
        for j, match in enumerate(matches):
            distances[j] = match[2]
        max_distances[i] = np.max(distances)
        min_distances[i] = np.min(distances)
        mean_distances[i] = np.mean(distances)
        median_distances[i] = np.median(distances)
        top1_distances[i] = distances[0]

    print(f'max_distances: {max_distances}')
    print(f'min_distances: {min_distances}')
    print(f'mean_distances: {mean_distances}')
    print(f'median_distances: {median_distances}')

    # plot
    # plt.plot(max_distances, label='max_distances')
    plt.plot(min_distances, label='min_distances')
    # plt.plot(mean_distances, label='mean_distances')
    # plt.plot(median_distances, label='median_distances')
    plt.plot(top1_distances, label='top1_distances')
    plt.legend()
    plt.show()

def readImage(image_path, size, score=0.0, distance=0.0):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, size)
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (5, size[0]-5) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method
    image = cv2.putText(image, str(distance), org, font, fontScale, color, thickness, cv2.LINE_AA) 
    return image

def visualize_image_pairs(args):
    image_pairs = readImagePairs(args.query, args.mapping, args.image_pairs_file)
    
    statistics_analysis(image_pairs)

    nFailure = 0
    nSuccess = 0
    for query, matches in tqdm.tqdm(image_pairs.items()):
        distances = np.zeros((20))
        for i in range(len(matches)):
            distances[i] = matches[i][2]
        distances = (distances < 10.0)
        if distances[0:args.topk].any():
            nSuccess += 1
        else:
            nFailure += 1
            # print(f'distances {distances}')
            if args.display:
                im_query = readImage(query, (1280, 1280))
                row_1 = []
                row_2 = []
                for i in range(len(matches)):
                    im_match = readImage(matches[i][0], (640, 640), matches[i][1], matches[i][2])
                    if i < len(matches) / 2:
                        row_1.append(im_match)
                    else:
                        row_2.append(im_match)
                im_row1 = cv2.hconcat(row_1)
                im_row2 = cv2.hconcat(row_2)
                im_matches = cv2.vconcat([im_row1, im_row2])
                im_result = cv2.hconcat([im_query, im_matches])
                cv2.namedWindow("image retrieval", cv2.WINDOW_NORMAL)
                cv2.imshow("image retrieval", im_result)
                cv2.waitKey(0)
    print(f'nSuccess: {nSuccess} nFailure: {nFailure} precision {float(nSuccess)/float(len(image_pairs))} @top_{args.topk}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='visualize image pairs')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--image_pairs_file', required=True, type=str, help='path to the image pairs file')
    parser.add_argument('--mapping', required=True, type=str, help='path to the mapping kapture root directory')
    parser.add_argument('--query', required=True, type=str, help='path to the query kapture root directory')
    parser.add_argument('--topk', type=int, default=20, help='top K retrieval images')
    parser.add_argument('--display', action='store_true', default=False, help='display results default False')


    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))

    visualize_image_pairs(args)