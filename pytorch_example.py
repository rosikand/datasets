"""
File: pytorch_example.py
------------------
Simple script which loads all of the image raw urls for a
dataset folder and inputs them into a PyTorch DataLoader. 
"""


import requests
import torch
from torch.utils.data import Dataset
import rsbox 
from rsbox import ml, misc
import torchvision 
from glob import glob
import pdb



def load_text_file_as_list(url):
    """
    Loads a remote text file specified at the url as a list of strings.
    """

    req = requests.get(url)
    req = req.text
    urls = req.split("\n")
    urls = urls[:-1] if len(urls[-1]) == 0 else urls
    return urls




class GithubStreamingDataset(Dataset):
    """
    Dataset class for streaming image data from a remote GH directory. 
    Specify a list of urls to the images in the directory or a url to a text file
    containing a list of urls to the images in the directory. 
    Arguments:
        dirpath: path to directory containing images
        resize: tuple of (height, width) to resize images to. None means no resizing.
        normalize: whether to normalize images to [0, 1]
        extension: extension of images in directory
        crop: (int) size to center crop images to. None means no cropping.
    """
    def __init__(self, list_of_urls, is_text_file=False, resize=None, normalize=True, extension="jpg", crop=None, neg_one_normalize=False, repeat_graysclale=False, dataset_multiplier=1):
        
        self.img_paths = list_of_urls if not is_text_file else load_text_file_as_list(list_of_urls)
        self.to_resize = True if resize is not None else False
        self.size = resize if self.to_resize else None
        self.normalize = normalize
        self.crop = crop
        self.neg_one_normalize = neg_one_normalize
        if self.neg_one_normalize and not self.normalize:
            raise ValueError("neg_one_normalize is set to True, but normalize is set to False. This is not allowed.")
        self.repeat_grayscale = repeat_graysclale
        self.dataset_multiplier = dataset_multiplier
        
    def __getitem__(self, index):
        sample_file = self.img_paths[index % len(self.img_paths)]
        sample = ml.get_img(sample_file, resize=self.to_resize, size=self.size)

        sample = torch.tensor(sample, dtype=torch.float)

        if self.repeat_grayscale:
            # makes grayscale images 3 channels to work with RGB models (e.g. u-net)
            if sample.shape[0] == 1:
                sample = torch.repeat_interleave(sample, 3, 0)
        
        if self.normalize:
            sample = sample / 255.0

        if self.neg_one_normalize and self.normalize:
            sample = sample * 2 - 1

        if self.crop is not None:
            sample = torchvision.transforms.CenterCrop(self.crop)(sample)

        return sample
        
    def __len__(self):
        return len(self.img_paths) * self.dataset_multiplier
        



sample_url = "https://raw.githubusercontent.com/rosikand/datasets/main/file_lists/ddpm-mini-dist.txt"

dataset = GithubStreamingDataset(sample_url, is_text_file=True, resize=(256, 256), normalize=True, crop=224, neg_one_normalize=False, repeat_graysclale=False, dataset_multiplier=1)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

for sample in dataloader:
    print(sample.shape)
    ml.plot(sample)
