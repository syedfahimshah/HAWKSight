import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from noise import pnoise2 
# ----------------------------------------------
#  Cloud and Shadow Mask Augmentation
# ----------------------------------------------
def add_cloud_shadow(image, intensity=0.7, shadow_intensity=0.4):
    """Apply cloud and shadow effects to an image"""
    img_array = np.array(image).astype(np.float32) / 255.0
    h, w, _ = img_array.shape
    
    # Create random cloud patterns using Perlin noise
    scale = random.uniform(0.02, 0.1)  # Cloud scale
    octaves = random.randint(2, 5)     # Noise complexity
    persistence = random.uniform(0.5, 0.8)
    
    cloud = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            cloud[i][j] = pnoise2(i * scale, 
                                 j * scale, 
                                 octaves=octaves, 
                                 persistence=persistence,
                                 repeatx=w,
                                 repeaty=h)
    
    # Normalize and apply intensity
    cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min())
    cloud_mask = np.clip(cloud * intensity, 0, 1)
    
    # Apply shadow effect (darker areas)
    shadow_mask = np.clip(cloud * shadow_intensity, 0, 0.6)
    
    # Randomly choose between cloud or shadow effect
    if random.random() > 0.5:
        # Cloud effect (lighten areas)
        img_array = img_array * (1 - cloud_mask[..., np.newaxis]) + cloud_mask[..., np.newaxis]
    else:
        # Shadow effect (darken areas)
        img_array = img_array * (1 - shadow_mask[..., np.newaxis])
    
    img_array = np.clip(img_array, 0, 1) * 255
    return Image.fromarray(img_array.astype(np.uint8))
#several data augumentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        #edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label
def randomCrop(image, label):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)
def randomRotation(image,label):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        #edge=edge.rotate(random_angle, mode)
    return image,label
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        #self.edges=[edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
        #            or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
       # self.edges=sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        #self.edge_transform = transforms.Compose([
        #    transforms.Resize((self.trainsize, self.trainsize)),
        #    transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        #edge=self.binary_loader(self.edges[index])
        image,gt =cv_random_flip(image,gt)
        image,gt=randomCrop(image, gt)
        image,gt=randomRotation(image, gt)
        image=colorEnhance(image)
        if random.random() < 0.4:
            image = add_cloud_shadow(image, 
                                    intensity=random.uniform(0.5, 0.9),
                                    shadow_intensity=random.uniform(0.3, 0.7))
        #edge=randomPeper(edge)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        #edge=self.edge_transform(edge)
        
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        #edges=[]
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            #edge= Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                #edges.append(edge_path)                #?????????????????????????????????????????
        self.images = images
        self.gts = gts
        #self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        #self.edges=[edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
        #            or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        #self.edges=sorted(self.edges)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        #self.edge_transform = transforms.ToTensor() 
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        #edge = self.binary_loader(self.edges[self.index])
        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, name,np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size