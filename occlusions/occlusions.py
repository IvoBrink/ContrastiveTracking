import os
import cv2
import pandas as pd
import numpy as np

from imageio.v3 import imread
from random import sample, choice


CAR = imread('car.png') / 255

class Occlusion:
    def __init__(self, data_path, name, occluded_objects=1, occlusion_frames=48, occlusion_type='black'):
        df = pd.read_table(f'{data_path}/label_02/{name}.txt', delimiter=' ', header=None)
        self.df = df[df[2] != "DontCare"]
        self.find_trajectories()
        self.load_images(f"{data_path}/image_02/{name}/{'{:06}.png'}")
        self.dims = np.asarray(self.imgs.shape[1:3])
        self.oo = occluded_objects
        self.ofs = occlusion_frames
        self.type = occlusion_type
    
    def find_trajectories(self):
        self.trajectories = {i: dict() for i in self.df[1].unique()}
        for f in range(self.df[0].max()):
            for _, (f, i, x1, y2, x2, y1) in self.df[self.df[0] == f].iloc[:, np.r_[0:2, 6:10]].iterrows():
                self.trajectories[i][f] = [x1, y1, x2, y2]
    
    def load_images(self, path):
        self.imgs = np.asarray([imread(path.format(f)) for f in range(self.df[0].max())]) / 255
    
    def select_objects(self):
        try: obs = sample(list(self.df[1].unique()), self.oo)
        except ValueError: obs = sample(list(self.df[1].unique()), 1)
        return obs
    
    @staticmethod
    def center_point(bbox):
        x1, y1, x2, y2 = bbox
        return np.asarray([(x2 + x1) / 2, (y2 + y1) / 2])
    
    def select_occlusion_trajectory(self, traj):
        if len(traj) < 3: return
        i = choice(range(1, len(traj) - 1))
        t_frames = sorted(traj.keys())
        velocity = choice([-1, 1]) * self.center_point(traj[t_frames[i+1]]) - self.center_point(traj[t_frames[i]])
        if np.linalg.norm(velocity) == 0: velocity = np.asarray([1, 0])
        self.velocity = velocity / np.linalg.norm(velocity) * self.dims / self.ofs * np.random.uniform(0.9, 1.1)
        self.frame = t_frames[i]

    def calc_margins(self, x, y):
        if self.type == 'car':
            cy, cx = CAR.shape[:2]
            scale = x/cx if cy/cx > y/x else y/cy # Scaling factor to ensure total occlusion
            return np.asarray(CAR.shape[:2][::-1]) / 2 * scale * 1.1
        return np.asarray([x, y]) / 2 * 1.1

    def calc_occlusion_dimensions(self, traj):
        x1, y1, x2, y2 = traj[self.frame]
        return self.calc_margins(x2 - x1, y1 - y2)
    
    def calc_occlusion(self, traj, frame, marg):
        center = self.center_point(traj[self.frame]) + self.velocity * (self.frame - frame)
        occl_bbox = np.asarray([center - marg, center + marg]).flatten()
        if self.type != 'car': occl_bbox += np.random.uniform(0, 10, (4,))
        return np.clip(occl_bbox.round(), 0, [*self.dims[::-1]] * 2).astype(int)

    def add_occlusion_to_frame(self, frame):
        if self.type == 'black':
            return np.zeros(3)
        if self.type == 'noise':
            y, x, c = frame.shape
            return np.random.uniform(size=(y, x, c))
        if self.type == 'car':
            y, x, _ = frame.shape
            car = cv2.resize(CAR, dsize=(x, y), interpolation=cv2.INTER_CUBIC) \
                  if x * y != 0 else np.zeros((y, x, 4))
            if self.velocity[0] > 0: car = np.flip(car, axis=1)
            frame[car[:,:,3] != 0] = car[:,:,:3][car[:,:,3] != 0]
            return frame
    
    def add_occlusion(self, traj):
        marg = self.calc_occlusion_dimensions(traj)
        for f in range(self.df[0].max()):
            x1, y1, x2, y2 = self.calc_occlusion(traj, f, marg)
            self.imgs[f,y1:y2,x1:x2] = self.add_occlusion_to_frame(self.imgs[f,y1:y2,x1:x2])
    
    def add_occlusions(self):
        for o in self.select_objects():
            traj = self.trajectories[o]
            self.select_occlusion_trajectory(traj)
            self.add_occlusion(traj)
    
    def add_bboxes(self, f):
        img = self.imgs[f]
        red = np.asarray([0, 0, 1])
        t = 2 # Line thickness x2
        for _, (x1, y2, x2, y1) in self.df[self.df[0] == f].iloc[:, 6:10].round().astype(int).iterrows():
            img[y1-t:y1+t, x1:x2] = red
            img[y2-t:y2+t, x1:x2] = red
            img[y2:y1, x1-t:x1+t] = red
            img[y2:y1, x2-t:x2+t] = red
        return img
    
    def save_video(self, bboxes=True):
        out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 24, self.dims[::-1].astype(int))
        for f in range(self.imgs.shape[0]):
            f = self.add_bboxes(f) if bboxes else self.imgs[f]
            out.write((f*255).astype('uint8'))
        out.release()


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