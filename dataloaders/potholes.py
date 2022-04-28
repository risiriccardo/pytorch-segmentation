from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class POTHOLESDataset(BaseDataSet):
    """
    potholes dataset
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.POTHOLES_palette
        super(POTHOLESDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, '/content/drive/MyDrive/pothole600')
        self.image_dir = os.path.join(self.root, 'training', 'rgb', '*.png')
        self.label_dir = os.path.join(self.root, 'training', 'label' '*.png')
        #self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.png')]
        #
        self.files = sorted(glob.glob(self.image_dir))
#
    def _load_data(self, index):
        image_id = self.files[index]
        #image_path = os.path.join(self.image_dir, image_id + '.png')
        #label_path = os.path.join(self.label_dir, image_id + '.png')
        #image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        #label = np.asarray(Image.open(label_path), dtype=np.int32) - 1 # from -1 to 149
        #return image, label, image_id

        useDir = "/".join(self.files[index].split('/')[:-2])
        name = self.files[index].split('/')[-1]

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'rgb', name)), cv2.COLOR_BGR2RGB)
        #tdisp_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'tdisp', name)), cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(useDir, 'label', name), cv2.IMREAD_ANYDEPTH)
        label[label > 0] = 1
        oriHeight, oriWidth, _ = rgb_image.shape

        # resize image to enable sizes divide 16 for AA-UNet, and to enable divide 32 for AA-RTFNet
        rgb_image = cv2.resize(rgb_image, self.use_size)
        #tdisp_image = cv2.resize(tdisp_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        rgb_image = rgb_image.astype(np.float32) / 255
        #tdisp_image = tdisp_image.astype(np.float32) / 255
        image = transforms.ToTensor()(rgb_image)
        #tdisp_image = transforms.ToTensor()(tdisp_image)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, tdisp images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return image, label, image_id
#
class POTHOLES(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = POTHOLESDataset(**kwargs)
        super(POTHOLES, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
