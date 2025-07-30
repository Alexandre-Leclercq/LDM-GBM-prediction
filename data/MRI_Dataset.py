"""
Author: Alexandre G. Leclercq
"""

import os
from typing import Literal, Optional
import pandas as pd
import nibabel as nib
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import functional as F


class MRI_Dataset(Dataset):

    def __init__(self, dataset: pd.DataFrame, dataset_path,
                 task: Literal["autoencoder", "generation", "conditional_generation", "only_pre", "only_treatment", "only_post", "only_gtv", "free_guidance_conditionnal_generation"],
                 data_augmentation: bool = False,
                 process_transform: Optional[transforms.Compose] = None,
                 normalization_transform: Optional[dict] = None) -> None:
        self.data = dataset

        # assert that only one type of transformation is provided
        # either a specific transformation for each patient or the same transformation for all patient
        if task == 'only_pre':
            assert all(elem in self.data.columns for elem in ['preMRI', 'GTV'])
        elif task == 'only_gtv':
            assert all(elem in self.data.columns for elem in ['GTV'])
        elif task == 'only_treatment':
            assert all(elem in self.data.columns for elem in ['treatment', 'GTV'])
        elif task == 'only_post':
            assert all(elem in self.data.columns for elem in ['postMRI', 'GTV'])
        elif task == 'autoencoder':
            assert all(elem in self.data.columns for elem in ['type', 'MRI_img', 'GTV'])
        elif task == 'generation':
            assert all(elem in self.data.columns for elem in ['preMRI', 'postMRI', 'GTV'])
        elif task == 'conditional_generation':
            assert all(elem in self.data.columns for elem in ['preMRI', 'treatment', 'postMRI', 'GTV'])
        elif task == 'free_guidance_conditionnal_generation':
            assert all(elem in self.data.columns for elem in ['preMRI', 'postMRI', 'GTV', 'class'])
        else:
            print(f'task: {task} not supported')
            raise NotImplementedError

        self.dataset_path = dataset_path
        self.task = task
        self.data_augmentation = data_augmentation
        self.process_transform = process_transform
        self.normalization_transform = normalization_transform

    def __len__(self) -> int:
        return len(self.data)

    def __load_img(self, patient: str, slice_: str, img_column_name: str, idx: int):
        img_path = os.path.join(self.dataset_path, 'data', patient, slice_, self.data.iloc[idx][img_column_name])
        return nib.load(img_path).get_fdata()

    def get_dataset_row(self, idx):
        return self.data.iloc[idx]

    def get_nifti_metadata(self, patient: str, img_column_name: str):
        # retrive the first row of patient to get retrieve and image and get the correspond nifti metadata.
        patient_row_image = self.data[self.data['patient'] == patient].iloc[0]
        img_path = os.path.join(self.dataset_path, 'data', str(patient), str(patient_row_image['slice']), patient_row_image[img_column_name])
        img = nib.load(img_path)
        return img.affine.copy(), img.header.copy()

    def __getitem__(self, idx) -> dict:
        """
        return a dictionary where each key correspond to a modality of our dataset
        """
        patient = str(self.data.iloc[idx]['patient'])
        slice_ = str(self.data.iloc[idx]['slice'])

        # sample if we operate a random data augmentation for this data sample
        # thus allowing us to apply the same data augmentation to all our modality
        sample_random_transform = random.random() > 0.5

        gtv = self.__load_img(patient, slice_, 'GTV', idx)

        # apply the same transform to gtv than to our others images modalities to stay coherent
        if self.process_transform is not None:
            gtv = self.process_transform(gtv)
        if self.data_augmentation and sample_random_transform:
            gtv = F.vertical_flip(gtv)
        gtv[gtv > 0] = 1

        if self.task == 'only_gtv':
            return {'gtv': gtv, 'patient': patient, 'slice': slice_}
        if self.task == 'autoencoder':
            mri_img = self.__load_img(patient, slice_, 'MRI_img', idx)
            if self.process_transform is not None:
                mri_img = self.process_transform(mri_img)
            if self.normalization_transform is not None:
                if self.data.iloc[idx]['type'] == 'preMRI':
                    mri_img = self.normalization_transform['preMRI'][str(patient)](mri_img)
                elif self.data.iloc[idx]['type'] == 'postMRI':
                    mri_img = self.normalization_transform['postMRI'][str(patient)](mri_img)
                else:
                    raise NotImplementedError
            if self.data_augmentation and sample_random_transform:
                mri_img = F.vertical_flip(mri_img)
            return {'mri': mri_img, 'gtv': gtv, 'patient': patient, 'slice': slice_}

        if self.task == 'only_pre':
            images_types = ['preMRI']
        elif self.task == 'only_treatment':
            images_types = ['treatment']
        elif self.task == 'only_post':
            images_types = ['postMRI']
        elif self.task == 'generation' or self.task == 'free_guidance_conditionnal_generation':
            images_types = ['preMRI', 'postMRI']
        elif self.task == 'conditional_generation':
            images_types = ['preMRI', 'treatment', 'postMRI']
        else:
            raise NotImplementedError

        imgs = {}
        for images_type in images_types:
            imgs[images_type] = self.__load_img(patient, slice_, images_type, idx)
            if self.process_transform is not None:
                imgs[images_type] = self.process_transform(imgs[images_type])
            if self.normalization_transform is not None:
                imgs[images_type] = self.normalization_transform[str(images_type)][str(patient)](imgs[images_type])
            if self.data_augmentation and sample_random_transform:
                imgs[images_type] = F.vertical_flip(imgs[images_type])
        if self.task == 'free_guidance_conditionnal_generation':
            return {**imgs, 'gtv': gtv, 'class': int(self.data.iloc[idx]['class']), 'patient': patient, 'slice': slice_}
        else:
            return {**imgs, 'gtv': gtv, 'patient': patient, 'slice': slice_}

