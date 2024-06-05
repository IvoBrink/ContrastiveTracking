import os

from random import choice
from occlusions import Occlusion
from sys import argv
from tqdm import tqdm


if __name__ == "__main__":
    occ_type = argv[1] if len(argv) > 1 and argv[1] in ['black', 'noise', 'car'] else 'car'
    data_path = '../permatrack/data/kitti_tracking/data_tracking_image_2/training'
    label_path = '../permatrack/data/kitti_tracking/label_02_train_half'
    for vid in os.listdir(os.path.join(data_path, 'image_02')):
        for i in tqdm(range(10)):
            occ = Occlusion(data_path, vid,
                            label_path=label_path,
                            occluded_objects=choice(range(8)),
                            occlusion_frames=90,
                            occlusion_type=occ_type)
            occ.add_occlusions()
            occ.save_kitti('../permatrack/data/occluded', vid, i)