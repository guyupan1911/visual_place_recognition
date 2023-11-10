import os

import numpy as np

import kapture
from kapture.io.csv import kapture_from_dir, table_from_file, get_all_tar_handlers
from kapture.io.records import get_image_fullpath

def readImagePairs(query, mapping, image_pairs_file):
    mapping_tar_handlers = get_all_tar_handlers(mapping,
                                                mode={kapture.Keypoints: 'r',
                                                kapture.Descriptors: 'r',
                                                kapture.GlobalFeatures: 'r',
                                                kapture.Matches: 'a'}) 
    kdata_mapping = kapture_from_dir(mapping, image_pairs_file,
                                     skip_list=[kapture.Keypoints,
                                                kapture.Descriptors],
                                     tar_handlers=mapping_tar_handlers)
    query_tar_handlers = get_all_tar_handlers(query,
                                              mode={kapture.Keypoints: 'r',
                                              kapture.Descriptors: 'r',
                                              kapture.GlobalFeatures: 'r',
                                              kapture.Matches: 'a'}) 
    kdata_query = kapture_from_dir(query, image_pairs_file,
                                   skip_list=[kapture.Keypoints,
                                              kapture.Descriptors],
                                   tar_handlers=query_tar_handlers)
    if kdata_query.records_camera is not None:
        query_images_set = set(kdata_query.records_camera.data_list())
    else:
        query_images_set = None
    if kdata_mapping.records_camera is not None:
        map_images_set = set(kdata_mapping.records_camera.data_list())
    else:
        map_images_set = None

    if kdata_query.trajectories is not None:
        query_trajectories = kdata_query.trajectories
    if kdata_mapping.trajectories is not None:
        map_trajectories = kdata_mapping.trajectories

    image_pairs = {} # {key: query_image_path: [(map_image_path, score, distance)]}
    last_query_time = None
    with open(image_pairs_file, 'r') as fid:
        table = table_from_file(fid)
        for query_name, map_name, score in table:
            if query_images_set is not None and query_name not in query_images_set:
                continue
            if map_images_set is not None and map_name not in map_images_set:
                continue

            query_name = get_image_fullpath(query, query_name)
            map_name = get_image_fullpath(mapping, map_name)
            image_pairs.setdefault(query_name, []).append((map_name, score))

    # remove redundant matches
    reduced_image_pairs = {}
    first = True
    last_query_pose = None
    for query, matches in image_pairs.items():
        
        query_time = int(os.path.basename(query).split('.')[0])
        query_pose = query_trajectories.intermediate_pose(query_time, "car", 2e8)
        if query_pose is None:
            continue
        
        if first == True:
            first = False
        else:
            if np.linalg.norm(query_pose.t - last_query_pose.t) < 2.0:
                continue    
            for match in matches:
                mactch_time = int(os.path.basename(match[0]).split('.')[0])
                match_pose = map_trajectories.intermediate_pose(mactch_time, "car", 2e8)
                if match_pose is None:
                    del reduced_image_pairs[query]
                distance = np.absolute(query_pose.t - match_pose.t)
                reduced_image_pairs.setdefault(query, []).append([match[0], match[1], distance])

        last_query_pose = query_pose
    
    return reduced_image_pairs