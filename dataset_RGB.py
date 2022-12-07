import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
#import time
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        image_files = sorted(os.listdir(os.path.join(rgb_dir, 'image')))
        matte_files = sorted(os.listdir(os.path.join(rgb_dir, 'matte')))
        trimap_files = sorted(os.listdir(os.path.join(rgb_dir, 'trimap')))

        self.image_filenames = [os.path.join(rgb_dir, 'image', x)  for x in image_files if is_image_file(x)]
        self.matte_filenames = [os.path.join(rgb_dir, 'matte', x) for x in matte_files if is_image_file(x)]
        self.trimap_filenames = [os.path.join(rgb_dir, 'matte', x) for x in trimap_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.matte_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        image_path = self.image_filenames[index_]
        matte_path = self.matte_filenames[index_]
        trimap_path = self.trimap_filenames[index_]

        image_img = Image.open(image_path)
        matte_img = Image.open(matte_path)
        trimap_img = Image.open(trimap_path)

        image_img = TF.to_tensor(image_img)
        matte_img = TF.to_tensor(matte_img)
        trimap_img = TF.to_tensor(trimap_img)


        #filename = os.path.splitext(os.path.split(matte_img)[-1])[0]
        
        return matte_img, image_img,trimap_img

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename
