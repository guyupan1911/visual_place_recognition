import argparse
import logging
from functools import lru_cache
from typing import Optional
import torch
from tqdm import tqdm
import os

import kapture_localization.utils.logging
from kapture_localization.matching import MatchPairNnTorch
from kapture_localization.utils.pairsfile import get_pairs_from_file

import kapture
from kapture.io.csv import kapture_from_dir, get_all_tar_handlers

logger = logging.getLogger('visualize image pairs')

def visualize_feature_matches(args):
    mapping_tar_handlers = get_all_tar_handlers(args.mapping,
                                                mode={kapture.Keypoints: 'r',
                                                kapture.Descriptors: 'r',
                                                kapture.GlobalFeatures: 'r',
                                                kapture.Matches: 'r'}) 
    kdata_mapping = kapture_from_dir(args.mapping, args.image_pairs_file,
                                     skip_list=[],
                                     tar_handlers=mapping_tar_handlers)
    query_tar_handlers = get_all_tar_handlers(args.query,
                                              mode={kapture.Keypoints: 'r',
                                              kapture.Descriptors: 'r',
                                              kapture.GlobalFeatures: 'r',
                                              kapture.Matches: 'r'}) 
    kdata_query = kapture_from_dir(args.query, args.image_pairs_file,
                                   skip_list=[],
                                   tar_handlers=query_tar_handlers)


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

    visualize_feature_matches(args)