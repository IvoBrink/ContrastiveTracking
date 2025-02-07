import os
import cv2
import pandas as pd
import numpy as np

from imageio.v3 import imread
from random import sample, choice
from PIL import Image


PATH = 'data/cars/processed/'
CARS = [imread(f"{PATH}{car}") / 255 for car in os.listdir(PATH)]


class Occlusion:
    def __init__(self, data_path, name, label_path=None, occluded_objects=1, occlusion_frames=48, dataset='kitti', occlusion_type='black', save_annotations=False):
        self.shift = 1 if dataset == 'mot' else 0
        if save_annotations and dataset != 'mot':
            raise NotImplementedError(f"Saving annotations only supported for MOT dataset. Support for {dataset} not available.")
        self.save_annot = save_annotations
        try:
            getattr(self, f"load_{dataset}")(data_path, name, label_path)
        except AttributeError:
            raise AttributeError(f"Dataset {dataset} not supported. Available datasets are 'kitti' and 'mot'")
        self.find_trajectories(dataset)
        self.dims = np.asarray(self.imgs.shape[1:3])
        self.oo = occluded_objects
        self.ofs = occlusion_frames # Approximate number of frames the occlusion is on screen
        self.type = occlusion_type # From 'black', 'noise', and 'car'
    
    def load_kitti(self, data_path, name, label_path):
        label_path = f'{data_path}/label_02/{name}.txt' if label_path is None else f'{label_path}/{name}.txt'
        df = pd.read_table(label_path, delimiter=' ', header=None)
        self.df = df[df[2] != "DontCare"].iloc[:, np.r_[:2, 6:10]]
        self.load_images(f"{data_path}/image_02/{name}/{'{:06}.png'}")
    
    def load_mot(self, data_path, name, label_path):
        label_path = f'{data_path}/{name}/gt/gt.txt' if label_path is None else f'{label_path}/gt.txt'
        df = pd.read_csv(label_path, header=None).iloc[:, :6]
        df[4] += df[2]
        df[5] += df[3]
        self.df = df
        self.load_images(f"{data_path}/{name}/img1/{'{:06}.jpg'}")
        if self.save_annot:
            self.annot = pd.read_csv(label_path, header=None)
    
    def load_images(self, path):
        self.imgs = np.asarray([imread(path.format(f + self.shift))
                                for f in range(self.df[0].max() + 1 - self.shift)],
                               dtype=np.float16) / 255
    
    # Calculate the positions of every object at every frame
    def find_trajectories(self, dataset):
        self.trajectories = {i: dict() for i in self.df[1].unique()}
        for f in range(self.df[0].max() + 1 - self.shift):
            for _, (i, x1, y2, x2, y1) in self.df[self.df[0] == f + self.shift].iloc[:, 1:].iterrows():
                self.trajectories[i][f] = [x1, y1, x2, y2]
    
    def add_occlusions(self):
        for o in self.select_objects():
            self.fail = False
            traj = self.trajectories[o]
            self.select_occlusion_trajectory(traj)
            if self.fail: continue
            self.add_occlusion(traj)
    
    def select_objects(self):
        try: obs = sample(list(self.df[1].unique()), self.oo)
        except ValueError: obs = sample(list(self.df[1].unique()), 1)
        return obs
    
    def select_occlusion_trajectory(self, traj):
        # Occlude not during the first or last frame the object is visible
        if len(traj) < 3:
            self.fail = True
            return
        i = choice(range(1, len(traj) - 1))
        t_frames = sorted(traj.keys())
        
        # Move the occlusion with or against the path of the occluded object
        velocity = choice([-1, 1]) * self.center_point(traj[t_frames[i+1]]) - self.center_point(traj[t_frames[i]])
        if np.linalg.norm(velocity) == 0: # If the object does not move, the occlusion still should
            velocity = np.asarray([1, 0])
        self.velocity = velocity / np.linalg.norm(velocity) * self.dims / self.ofs * np.random.uniform(0.9, 1.1)
        self.frame = t_frames[i]
    
    @staticmethod
    def center_point(bbox):
        x1, y1, x2, y2 = bbox
        return np.asarray([(x2 + x1) / 2, (y2 + y1) / 2])
    
    def add_occlusion(self, traj):
        marg = self.calc_occlusion_dimensions(traj)
        if self.save_annot:
            obj = df[1].max() + 1
        for f in range(self.df[0].max() + 1 - self.shift):
            x1, y1, x2, y2 = self.calc_occlusion(traj, f, marg)
            if self.type == 'car': # Select only the pixels seen in frame
                self.cut_car(x1, y1, x2, y2)
            # Select only the valid part of the frame
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [*self.dims[::-1]] * 2)
            self.imgs[f,y1:y2,x1:x2] = self.add_occlusion_to_frame(self.imgs[f,y1:y2,x1:x2])
            if self.save_annot and np.prod(self.imgs[f,y1:y2,x1:x2].shape) != 0:
                self.annot = pd.concat([self.annot, pd.DataFrame(
                    [[f + self.shift, obj, x1, y1, x2-x1, y2-y1, -1, -1, -1]])])

    def calc_occlusion_dimensions(self, traj):
        x1, y1, x2, y2 = traj[self.frame]
        return self.calc_margins(x2 - x1, y1 - y2)

    def calc_margins(self, x, y):
        if self.type == 'car':
            self.CAR = choice(CARS)
            cy, cx = self.CAR.shape[:2]
            scale = x/cx if cy/cx > y/x else y/cy # Scaling factor to ensure total occlusion
            return np.asarray(self.CAR.shape[:2][::-1]) / 2 * scale * 1.1
        return np.asarray([x, y]) / 2 * 1.1
    
    def calc_occlusion(self, traj, frame, marg):
        center = self.center_point(traj[self.frame]) + self.velocity * (self.frame - frame)
        occl_bbox = np.asarray([center - marg, center + marg]).flatten()
        if self.type != 'car': # Add some noise to the size of the square
            occl_bbox += np.random.uniform(0, 10, (4,))
        return occl_bbox.round().astype(int)

    def cut_car(self, x1, y1, x2, y2):
        self.car = cv2.resize(self.CAR, dsize=(x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
        if self.velocity[0] > 0: # Orient the car backwards so its movement looks forward
            self.car = np.flip(self.car, axis=1)
        
        coords = np.stack(np.meshgrid(np.linspace(x1, x2 - 1, x2 - x1), np.linspace(y1, y2 - 1, y2 - y1)))
        coords = np.moveaxis(coords, (0, 1, 2), (-1, 0, 1))
        
        # Find which coordinates lie in the frame
        mask = np.logical_and(coords >= [0, 0], coords <= self.dims[::-1]).all(axis=-1)
        mask = np.tile(np.expand_dims(mask, -1), (1, 1, 4))
        self.car = self.car[mask].reshape(mask.sum(axis=0).max(), mask.sum(axis=1).max(), 4)

    def add_occlusion_to_frame(self, frame):
        y, x, c = frame.shape
        if self.type == 'noise':
            return np.random.uniform(size=(y, x, c))
        if self.type == 'car':
            # Ensure that car fragment matches selected fragment
            self.car = cv2.resize(self.car, dsize=(x, y), interpolation=cv2.INTER_CUBIC) \
                  if x * y != 0 else np.zeros((y, x, 4))
            frame[self.car[:,:,3] != 0] = self.car[:,:,:3][self.car[:,:,3] != 0]
            return frame
        return np.zeros(3)
    
    def save_video(self, path, bboxes=True, line_thickness=4):
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','J','P','G'), 24, self.dims[::-1].astype(int))
        for f in range(self.imgs.shape[0]):
            f = self.add_bboxes(f, line_thickness) if bboxes else self.imgs[f]
            out.write((f[:,:,::-1]*255).astype('uint8'))
        out.release()
    
    def add_all_bboxes(self, line_thickness=4):
        for f in range(self.imgs.shape[0]):
            self.imgs[f] = self.add_bboxes(f, line_thickness)
    
    def add_bboxes(self, f, line_thickness):
        img = self.imgs[f]
        red = np.asarray([1, 0, 0])
        t = line_thickness // 2
        df = self.df
        if self.save_annot:
            df = self.annot.iloc[:, :6]
            df[4] += df[2]
            df[5] += df[3]
        for _, (x1, y2, x2, y1) in df[df[0] == f + self.shift].iloc[:, 2:].round().astype(int).iterrows():
            img[y1-t:y1+t, x1:x2] = red
            img[y2-t:y2+t, x1:x2] = red
            img[y2:y1, x1-t:x1+t] = red
            img[y2:y1, x2-t:x2+t] = red
        return img
    
    def save_kitti(self, path, name, id_):
        path = os.path.join(path, f"{name}.{id_:03}")
        for i, img in enumerate((self.imgs * 255).astype(np.uint8)):
            if not os.path.exists(path):
                os.makedirs(path)
            Image.fromarray(img).save(os.path.join(path, f"{i:06}.png"))
    
    def save_mot(self, path, name):
        path = os.path.join(path, name, 'img1')
        for i, img in enumerate((self.imgs * 255)[:50].astype(np.uint8)):
            if not os.path.exists(path):
                os.makedirs(path)
            Image.fromarray(img).save(os.path.join(path, f"{i+1:06}.jpg"))
        if self.save_annot:
            path = os.path.join(os.path.dirname(path), 'gt')
            if not os.path.exists(path):
                os.makedirs(path)
            self.annot.to_csv(os.path.join(path, "gt.txt"), index=False, header=False)

def read_and_occlude(path, video, occluded_objects=4, occlusion_frames=90, occlusion_type='car'):
    occ = Occlusion(
        path,
        video,
        occluded_objects=occluded_objects,
        occlusion_frames=occlusion_frames,
        occlusion_type=occlusion_type
    )
    occ.add_occlusions()
    return occ.imgs