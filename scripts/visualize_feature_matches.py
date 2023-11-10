import argparse
import logging
from functools import lru_cache
from typing import Optional
import torch
from tqdm import tqdm
import os

import numpy as np
import cv2

import kapture_localization.utils.logging
from kapture_localization.utils.pairsfile import get_pairs_from_file

import kapture
from kapture.io.csv import kapture_from_dir, get_all_tar_handlers
from kapture.io.records import get_image_fullpath
from kapture.io.features import get_keypoints_fullpath, image_keypoints_from_file, get_descriptors_fullpath, image_descriptors_from_file

from utils.pairsfile import readImagePairs
from utils.matching import match_descriptors

logger = logging.getLogger('visualize image pairs')

def draw_keypoints(raw_image, kpts_numpy):
    kpts = [cv2.KeyPoint(kpts_numpy[i][0], \
                         kpts_numpy[i][1], 1, \
                         response = kpts_numpy[i][2]) \
            for i in range(kpts_numpy.shape[0])]
    raw_image = cv2.drawKeypoints(raw_image, kpts, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return raw_image

def visualize_feature_matches(query_name: str, kapture_query_dir: str, \
                              match_name: str, kapture_match_dir: str):
    # load and resize image
    query_image_full_path = get_image_fullpath(kapture_query_dir, query_name)
    match_image_full_path = get_image_fullpath(kapture_match_dir, match_name)
    im_query = cv2.imread(query_image_full_path)
    im_query = cv2.resize(im_query, (640, 640))
    im_match = cv2.imread(match_image_full_path)
    im_match = cv2.resize(im_match, (640, 640)) 
    # print(f'query_image_full_path: {query_image_full_path}\nmatch_image_full_path: {match_image_full_path}')

    # load and draw keypoints
    query_keypoints_full_path = get_keypoints_fullpath("superpoint_v1", kapture_query_dir, query_name)
    query_kpts_array = image_keypoints_from_file(query_keypoints_full_path, np.float64, 3)
    query_kpts = [cv2.KeyPoint(query_kpts_array[i][0], \
                         query_kpts_array[i][1], 1, \
                         response = query_kpts_array[i][2]) \
                  for i in range(query_kpts_array.shape[0])]

    match_keypoints_full_path = get_keypoints_fullpath("superpoint_v1", kapture_match_dir, match_name)
    match_kpts_array = image_keypoints_from_file(match_keypoints_full_path, np.float64, 3)
    match_kpts = [cv2.KeyPoint(match_kpts_array[i][0], \
                                match_kpts_array[i][1], 1, \
                                response = match_kpts_array[i][2]) \
                  for i in range(match_kpts_array.shape[0])]    # print(f'query kpts: {query_kpts.shape}, match kpts: {match_kpts.shape}')

    # load and match descriptors
    query_descriptors_path = get_descriptors_fullpath("superpoint_v1", kapture_query_dir, query_name)
    query_descriptors_array = image_descriptors_from_file(query_descriptors_path, np.float32, 256)
    match_descriptors_path = get_descriptors_fullpath("superpoint_v1", kapture_match_dir, match_name)
    match_descriptors_array = image_descriptors_from_file(match_descriptors_path, np.float32, 256)
    # print(f'query desc: {query_descriptors_array.shape}, match desc: {match_descriptors_array.shape}')

    # NN Matcher
    matches_array = match_descriptors(query_descriptors_array, match_descriptors_array)
    matches = [cv2.DMatch(int(matches_array[i][0]), int(matches_array[i][1]), matches_array[i][2]) \
               for i in range(matches_array.shape[0])]
    # print(f'matches type: {type(matches_array)} {matches_array.shape} {matches_array[0]}') 

    # display
    cv2.namedWindow('feature_matches', cv2.WINDOW_NORMAL)
    im_show = cv2.hconcat([im_query, im_match])
    cv2.drawMatches(im_query, query_kpts, im_match, match_kpts, matches, im_show, flags=2)
    cv2.imshow('feature_matches', im_show)
    cv2.waitKey(0)

    return len(matches)


def visualize_feature_matches_command_line(args):
    
    kdata_mapping = kapture_from_dir(args.mapping, args.image_pairs_file, skip_list=[])
    kdata_query = kapture_from_dir(args.query, args.image_pairs_file, skip_list=[])

    print(f'keypoints {next(iter(kdata_mapping.keypoints))}')

    image_pairs = readImagePairs(args.query, args.mapping, args.image_pairs_file)
    logger.info(f'image_pairs size: {len(image_pairs) * len(image_pairs.get(next(iter(image_pairs))))}')
    for query, matches in image_pairs.items():
        for match in matches:
            nfeature_matches = visualize_feature_matches(os.path.join('images', os.path.basename(query)), args.query, \
                                      os.path.join('images', os.path.basename(match[0])), args.mapping)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='visualize feature matches')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--image-pairs-file', required=True, type=str, help='path to the image pairs file')
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

    visualize_feature_matches_command_line(args)