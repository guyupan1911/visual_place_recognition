import os
import tqdm
import numpy as np

import kapture  # noqa: E402
from kapture.io.csv import kapture_from_dir, get_all_tar_handlers  # noqa: E402
from kapture.io.csv import get_feature_csv_fullpath, global_features_to_file  # noqa: E402
from kapture.io.records import get_image_fullpath  # noqa: E402
from kapture.io.features import get_global_features_fullpath, image_global_features_to_file  # noqa: E402
from kapture.io.features import global_features_check_dir  # noqa: E402

import seqslam

def extract_kapture_global_features(kapture_root_path: str, global_features_type: str):
    """ Extract seqslam features on a given dataset.
    """
    print(f'loading {kapture_root_path}')
    with get_all_tar_handlers(kapture_root_path,
                              mode={kapture.Keypoints: 'r',
                                    kapture.Descriptors: 'r',
                                    kapture.GlobalFeatures: 'a',
                                    kapture.Matches: 'r'}) as tar_handlers:
        kdata = kapture_from_dir(kapture_root_path, None,
                                 skip_list=[kapture.Keypoints,
                                            kapture.Descriptors,
                                            kapture.Matches,
                                            kapture.Points3d,
                                            kapture.Observations],
                                 tar_handlers=tar_handlers)
        root = get_image_fullpath(kapture_root_path, image_filename=None)
        assert kdata.records_camera is not None
        imgs = [image_name for _, _, image_name in kapture.flatten(kdata.records_camera)]
        if kdata.global_features is None:
            kdata.global_features = {}

        if global_features_type in kdata.global_features:
            imgs = [image_name
                    for image_name in imgs
                    if image_name not in kdata.global_features[global_features_type]]
        if len(imgs) == 0:
            print('All global features are already extracted')
            return

        net = seqslam.SeqSLAM()

        print('writing extracted global features')
        os.umask(0o002)
        gfeat_dtype = np.float32
        gfeat_dsize = 4096
        if global_features_type not in kdata.global_features:
            kdata.global_features[global_features_type] = kapture.GlobalFeatures(global_features_type, gfeat_dtype,
                                                                                 gfeat_dsize, 'L2')
            global_features_config_absolute_path = get_feature_csv_fullpath(kapture.GlobalFeatures,
                                                                            global_features_type,
                                                                            kapture_root_path)
            global_features_to_file(global_features_config_absolute_path, kdata.global_features[global_features_type])
        else:
            assert kdata.global_features[global_features_type].dtype == gfeat_dtype
            assert kdata.global_features[global_features_type].dsize == gfeat_dsize
            assert kdata.global_features[global_features_type].metric_type == 'L2'
        for image_name in tqdm.tqdm(imgs):
            full_image_name = get_image_fullpath(kapture_root_path, image_name)
            global_feature = net.run(full_image_name)
            global_feature_fullpath = get_global_features_fullpath(global_features_type, kapture_root_path, image_name)
            assert global_feature.shape == (gfeat_dsize,)
            image_global_features_to_file(global_feature_fullpath, global_feature)
            kdata.global_features[global_features_type].add(image_name)
            del global_feature

        if not global_features_check_dir(kdata.global_features[global_features_type], global_features_type,
                                         kapture_root_path):
            print('global feature extraction ended successfully but not all files were saved')
        else:
            print('Features extracted.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract SeqSLAM global features')
    parser.add_argument('--kapture-root', type=str, required=True, help='path to kapture root directory')
    args = parser.parse_args()

    # Evaluate
    res = extract_kapture_global_features(args.kapture_root, "SeqSLAM")
