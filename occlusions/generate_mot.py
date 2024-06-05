import os

from random import choice
from occlusions import Occlusion
from sys import argv
from tqdm import tqdm


if __name__ == "__main__":
    occ_type = argv[1] if len(argv) > 1 and argv[1] in ['black', 'noise', 'car'] else 'car'
    save_annot = True if len(argv) > 2 and argv[2] == 'annot' else False
    data_path = '../permatrack/data/mot17/train'
    save_path = f'../permatrack/data/mot_occluded_{occ_type}'
    if save_annot:
        save_path += "_annot/train"
    for vid in tqdm(os.listdir(data_path)):
        occ = Occlusion(data_path, vid,
                        occluded_objects=choice(range(8)),
                        occlusion_frames=choice(range(70, 120)),
                        occlusion_type=occ_type,
                        dataset='mot',
                        save_annotations=save_annot)
        occ.add_occlusions()
        occ.save_mot(save_path, vid)