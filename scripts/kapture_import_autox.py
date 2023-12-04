#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# Copyright 2016-present AutoX. Under BSD 3-clause license
"""
Script to import an autox dataset into a kapture.

The AutoX dataset contains

For each sequence, the recorded data is stored in the following structure:
├── KeyFrameData
├── distorted_images ...
├── undistorted_images
│   ├── cam0
│   └── cam1
├── GNSSPoses.txt
├── Transformations.txt
├── imu.txt
├── result.txt
├── septentrio.nmea
└── times.txt

The calibration folder has the following structure:
├── calib_0.txt
├── calib_1.txt
├── calib_stereo.txt
├── camchain.yaml
├── undistorted_calib_0.txt
├── undistorted_calib_1.txt
└── undistorted_calib_stereo.txt
"""

import argparse
import logging
import os
import os.path as path
import re
import math
import numpy as np
import quaternion
from glob import glob
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
# kapture
# import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.utils.logging import getLogger
from kapture.converter.nmea.import_nmea import extract_gnss_from_nmea

logger = getLogger()

RIG_ID = "car"


def load_gnss_poses_file(poses_file_path: str,
                         sensor_id: str) -> kapture.Trajectories:
    """ load GNSSPoses.txt to a kapture trajectories """
    pose_table = np.loadtxt(poses_file_path, delimiter=',', skiprows=1)
    timestamps_ns = (pose_table[:, 0]).astype(int)
    poses = pose_table[:, 1:8].astype(float)
    trajectories = kapture.Trajectories()
    for timestamp_ns, (tx, ty, tz, qx, qy, qz, qw) in zip(timestamps_ns, poses):
        timestamp_ns = int(timestamp_ns)
        car_from_world = kapture.PoseTransform(r=[qw, qx, qy, qz],
                                               t=[tx, ty, tz])
        trajectories[timestamp_ns, sensor_id] = car_from_world
    return trajectories


def import_autox_trajectory(poses_file_path: str) -> kapture.Trajectories:
    """
    imports trajectories.
    Using the globally optimized poses (inside GNSSPoses.txt),
    and transforming them using the chain of matrix multiplication from the Transformations.txt

    :param poses_file_path: full path to GNSSPoses.txt (name included)
    :param transformations_file_path: full path to Transformations.txt (name included)
    :return:
    """
    logger.info('importing images')
    # load gnss optimized poses
    trajectories = None
    if path.isfile(poses_file_path):
        trajectories = load_gnss_poses_file(poses_file_path, sensor_id=RIG_ID)

    return trajectories


def import_autox_images(
    recording_dir_path: str,
    kapture_dir_path: str,
    shot_id_to_timestamp: Dict[str, int],
    sensors: kapture.Sensors,
    images_import_method: TransferAction,
) -> kapture.RecordsCamera:
    """
    imports and copy image files.

    :param recording_dir_path:
    :param kapture_dir_path:
    :param shot_id_to_timestamp:
    :param sensors:
    :param images_import_method:
    :return:
    """
    kapture_images = kapture.RecordsCamera()
    logger.info('importing images ...')
    season_image_dir_path = recording_dir_path
    for shot_id, timestamp_ns in shot_id_to_timestamp.items():
        for ts in timestamp_ns:
            image_file_name = path.join(shot_id, f'{ts}.png')
            # check image file is available
            if not path.isfile(path.join(season_image_dir_path,
                                         image_file_name)):
                logger.warning(
                    f'image file is missing (and ignored) : {image_file_name}')
                # just throw it away
                continue
            kapture_images[ts, shot_id] = image_file_name

    filename_list = [f for _, _, f in kapture.flatten(kapture_images)]
    import_record_data_from_dir_auto(
        source_record_dirpath=season_image_dir_path,
        destination_kapture_dirpath=kapture_dir_path,
        filename_list=filename_list,
        copy_strategy=images_import_method)
    return kapture_images


def load_times_ids(times_file_path: str) -> Dict[int, int]:
    """ load times.txt file to a dict linking shot_id to timestamp in nanoseconds """
    table = np.loadtxt(times_file_path, delimiter=',', skiprows=1, dtype=str)
    shots_ids = table[:, 0].astype(str)
    timestamps_ns = table[:, 1].astype(int)
    shot_id_to_timestamp = {}
    for shot_id, timestamp_ns in zip(shots_ids, timestamps_ns):
        timestamp_ns = int(timestamp_ns)
        shot_id_to_timestamp.setdefault(shot_id, []).append(timestamp_ns)

    return shot_id_to_timestamp


def load_autox_cameras(
        calibration_dir_path: str) -> (kapture.Sensors, kapture.Rigs):
    sensors = kapture.Sensors()
    fx = 2447.462158203125
    cx = 1085.9044189453125
    fy = 2438.612548828125
    cy = 1094.943359375
    w = 2160
    h = 2160
    sensors['left_0_n_12mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                              [w, h, fx, fy, cx, cy],
                                              name='left_0_n_12mm')
    sensors['left_225_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                               [w, h, fx, fy, cx, cy],
                                               name='left_225_n_6mm')
    sensors['left_270_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                               [w, h, fx, fy, cx, cy],
                                               name='left_270_n_6mm')
    sensors['left_315_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                               [w, h, fx, fy, cx, cy],
                                               name='left_315_n_6mm')
    sensors['right_0_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                              [w, h, fx, fy, cx, cy],
                                              name='right_0_n_6mm')
    sensors['right_0_n_12mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                               [w, h, fx, fy, cx, cy],
                                               name='right_0_n_12mm')
    sensors['right_45_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                               [w, h, fx, fy, cx, cy],
                                               name='right_45_n_6mm')
    sensors['right_90_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                               [w, h, fx, fy, cx, cy],
                                               name='right_90_n_6mm')
    sensors['right_135_n_6mm'] = kapture.Camera(kapture.CameraType.PINHOLE,
                                                [w, h, fx, fy, cx, cy],
                                                name='right_135_n_6mm')

    rigs = kapture.Rigs()
    rigs[RIG_ID, 'left_0_n_12mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'left_225_n_6mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'left_270_n_6mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'left_315_n_6mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'right_0_n_6mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'right_0_n_12mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'right_45_n_6mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'right_90_n_6mm'] = kapture.PoseTransform()
    rigs[RIG_ID, 'right_135_n_6mm'] = kapture.PoseTransform()

    return sensors, rigs


def import_autox_sequence(calibration_dir_path: str, recording_dir_path: str,
                          kapture_dir_path: str,
                          images_import_method: TransferAction,
                          force_overwrite_existing: bool):
    """
    converts an autox recorded sequence to kapture format.

    :param calibration_dir_path: path to input calibration directory
    :param recording_dir_path: path to input sequence directory
    :param kapture_dir_path: path to output kapture directory
    :param images_import_method:
    :param force_overwrite_existing:
    :return:
    """
    delete_existing_kapture_files(kapture_dir_path,
                                  force_erase=force_overwrite_existing)
    os.makedirs(kapture_dir_path, exist_ok=True)

    # sensors
    sensors, rigs = load_autox_cameras(
        calibration_dir_path=calibration_dir_path)
    imported_kapture = kapture.Kapture(sensors=sensors, rigs=rigs)

    # timestamps
    times_filename = path.join(recording_dir_path, 'camera_timestamps.txt')
    shot_id_to_timestamp = load_times_ids(times_filename)

    # images
    records_camera = import_autox_images(
        recording_dir_path=recording_dir_path,
        kapture_dir_path=kapture_dir_path,
        shot_id_to_timestamp=shot_id_to_timestamp,
        sensors=sensors,
        images_import_method=images_import_method)
    imported_kapture.records_camera = records_camera

    # trajectories from lidar localization
    poses_file_path = path.join(recording_dir_path, 'lidar_pose.csv')
    if path.isfile(poses_file_path):
        trajectories = import_autox_trajectory(poses_file_path=poses_file_path)
        imported_kapture.trajectories = trajectories

    # finally save the kapture csv files
    kapture_to_dir(kapture_dir_path, imported_kapture)


def import_autox_command_line() -> None:
    """
    Imports AutoX dataset and save them as kapture using the parameters given on the command line.
    It assumes images are undistorted.
    """
    parser = argparse.ArgumentParser(
        description='Imports autox dataset files to the kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v',
        '--verbose',
        nargs='?',
        default=logging.WARNING,
        const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help=
        'verbosity level (debug, info, warning, critical, ... or int value) [warning]'
    )
    parser_verbosity.add_argument('-q',
                                  '--silent',
                                  '--quiet',
                                  action='store_const',
                                  dest='verbose',
                                  const=logging.CRITICAL)
    parser.add_argument('-f',
                        '-y',
                        '--force',
                        action='store_true',
                        default=False,
                        help='Force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='input path to autox record directory')
    parser.add_argument(
        '-c',
        '--calibration',
        help=
        'input path to autox calibration. If not given, assumed to be alongside input.'
    )
    parser.add_argument(
        '--image_transfer',
        type=TransferAction,
        default=TransferAction.link_absolute,
        help=f'How to import images [link_absolute], '
        f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='output directory.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    recording_dir_path = path.abspath(args.input)
    calibration_dir_path = ""

    import_autox_sequence(calibration_dir_path=calibration_dir_path,
                          recording_dir_path=recording_dir_path,
                          kapture_dir_path=args.output,
                          images_import_method=args.image_transfer,
                          force_overwrite_existing=args.force)


if __name__ == '__main__':
    import_autox_command_line()
