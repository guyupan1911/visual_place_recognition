#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

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
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_from_file, get_all_tar_handlers
from kapture.io.features import get_descriptors_fullpath, get_matches_fullpath
from kapture.io.features import image_descriptors_from_file
from kapture.io.features import matches_check_dir, image_matches_to_file
from kapture.io.tar import TarCollection
from kapture.utils.Collections import try_get_only_key_from_collection
from utils.pairsfile import readImagePairs

logger = logging.getLogger('compute_matches')


@lru_cache(maxsize=50)
def load_descriptors(descriptors_type: str, input_path: str,
                     tar_handler: Optional[TarCollection],
                     image_name: str, dtype, dsize):
    """
    load a descriptor. this functions caches up to 50 descriptors

    :param descriptors_type: type of descriptors, name of the descriptors subfolder
    :param input_path: input path to kapture input root directory
    :param tar_handler: collection of preloaded tar archives
    :param image_name: name of the image
    :param dtype: dtype of the numpy array
    :param dsize: size of the numpy array
    """
    descriptors_path = get_descriptors_fullpath(descriptors_type, input_path, image_name, tar_handler)
    return image_descriptors_from_file(descriptors_path, dtype, dsize)


def compute_matches(query: str,
                    mapping: str,
                    descriptors_type: Optional[str],
                    pairsfile_path: str,
                    overwrite_existing: bool = False):
    """
    compute matches from descriptors. images to match are selected from a pairsfile (csv with name1, name2, score)

    :param query_root_path: input path to kapture query root directory
    :param mapping_root_path: input path to kapture mapping root directory
    :param descriptors_type: type of descriptors, name of the descriptors subfolder
    :param pairsfile_path: path to pairs file (csv with 3 fields, name1, name2, score)
    :type pairsfile_path: str
    """
    logger.info(f'compute_feature_matches. loading query: {query} loading mapping {mapping}')
    mapping_tar_handlers = get_all_tar_handlers(mapping,
                                                mode={kapture.Keypoints: 'r',
                                                kapture.Descriptors: 'r',
                                                kapture.GlobalFeatures: 'r',
                                                kapture.Matches: 'a'}) 
    kdata_mapping = kapture_from_dir(mapping, pairsfile_path,
                                     skip_list=[kapture.GlobalFeatures,
                                                kapture.Observations,
                                                kapture.Points3d],
                                     tar_handlers=mapping_tar_handlers)
    query_tar_handlers = get_all_tar_handlers(query,
                                              mode={kapture.Keypoints: 'r',
                                              kapture.Descriptors: 'r',
                                              kapture.GlobalFeatures: 'r',
                                              kapture.Matches: 'a'}) 
    kdata_query = kapture_from_dir(query, pairsfile_path,
                                   skip_list=[kapture.GlobalFeatures,
                                              kapture.Observations,
                                              kapture.Points3d],
                                   tar_handlers=query_tar_handlers)
    
    image_pairs = readImagePairs(query, mapping, pairsfile_path)
    logger.info(f'image pairs size: {len(image_pairs)}')

    compute_matches_from_loaded_data(query,
                                     mapping,
                                     kdata_query,
                                     kdata_mapping,
                                     query_tar_handlers,
                                     mapping_tar_handlers,
                                     descriptors_type,
                                     image_pairs,
                                     overwrite_existing)

def compute_matches_from_loaded_data(query_dir: str,
                                     mapping_dir: str,
                                     kdata_query: kapture.Kapture,
                                     kdata_mapping: kapture.Kapture,
                                     query_tar_handlers: Optional[TarCollection],
                                     mapping_tar_handlers: Optional[TarCollection],
                                     descriptors_type: Optional[str],
                                     image_pairs: dict,
                                     overwrite_existing: bool = False):
    os.umask(0o002)
    if descriptors_type is None:
        descriptors_type = try_get_only_key_from_collection(kdata_query.descriptors)
    assert descriptors_type is not None
    assert descriptors_type in kdata_query.descriptors
    keypoints_type = kdata_query.descriptors[descriptors_type].keypoints_type
    logger.info(f'keypoints type: {keypoints_type} descriptors type: {descriptors_type}')
    # print(type(kdata_query.descriptors[descriptors_type]))
    
    # matcher
    matcher = MatchPairNnTorch(use_cuda=torch.cuda.is_available())
    new_matches = kapture.Matches()

    logger.info('compute_matches. entering main loop...')
    hide_progress_bar = logger.getEffectiveLevel() > logging.INFO
    skip_count = 0
    for query, matches in tqdm(image_pairs.items(), disable=hide_progress_bar):
        for match in matches:
            query_name = os.path.join('images', os.path.basename(query))
            match_name = os.path.join('images', os.path.basename(match[0]))
            # skip existing matches
            if (not overwrite_existing) \
                    and (kdata_query.matches is not None) \
                    and keypoints_type in kdata_query.matches \
                    and ((query_name, match_name) in kdata_query.matches[keypoints_type]):
                new_matches.add(query_name, match_name)
                skip_count += 1
                continue
            
            if query_name not in kdata_query.descriptors[descriptors_type] \
                    or match_name not in kdata_mapping.descriptors[descriptors_type]:
                logger.warning(f'unable to find descriptors for image pair: \n\t{query_name} \n\t{match_name}')
                continue

            descriptor_query = load_descriptors(descriptors_type, query_dir, query_tar_handlers,
                                                query_name, kdata_query.descriptors[descriptors_type].dtype,
                                                kdata_query.descriptors[descriptors_type].dsize)
            descriptor_match = load_descriptors(descriptors_type, mapping_dir, mapping_tar_handlers,
                                                match_name, kdata_mapping.descriptors[descriptors_type].dtype,
                                                kdata_mapping.descriptors[descriptors_type].dsize)
            matches = matcher.match_descriptors(descriptor_query, descriptor_match)
            matches_path = get_matches_fullpath((query_name, match_name), keypoints_type, query_dir, query_tar_handlers)
            image_matches_to_file(matches_path, matches)
            new_matches.add(query_name, match_name)
    
    if not overwrite_existing:
        logger.debug(f'{skip_count} pairs were skipped because the match file already existed')
    if not matches_check_dir(new_matches, keypoints_type, query_dir, query_tar_handlers):
        logger.critical('matching ended successfully but not all files were saved')


def compute_matches_command_line():
    parser = argparse.ArgumentParser(
        description='Compute matches with nearest neighbors from descriptors.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--query', required=True, help=('path to query kapture root directory'))
    parser.add_argument('--mapping', required=True, help=('path to mapping kapture root directory'))

    parser.add_argument('--pairsfile-path', required=True, type=str,
                        help=('text file in the csv format; where each line is image_name1, image_name2, score '
                              'which contains the image pairs to match'))
    parser.add_argument('-ow', '--overwrite', action='store_true', default=False,
                        help='overwrite matches if they already exist.')
    parser.add_argument('-desc', '--descriptors-type', default=None, help='kapture descriptors type.')
    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_matches(args.query,
                    args.mapping,
                    args.descriptors_type,
                    args.pairsfile_path,
                    args.overwrite)


if __name__ == '__main__':
    compute_matches_command_line()
