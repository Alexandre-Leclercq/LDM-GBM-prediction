"""
Author: Alexandre G. Leclercq
"""
import os
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
import numpy as np
from typing import Literal, Optional, Tuple
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype, Lambda, Resize, CenterCrop
from data.MRI_Dataset import MRI_Dataset
import time


class MRIDataModule(L.LightningDataModule):

    def __init__(self, dataset_path: str,
                 manifest_filename: str,
                 batch_size: int,
                 task: Literal["autoencoder", "generation", "conditional_generation", "only_pre", "only_treatment", "only_post", "free_guidance_conditionnal_generation"],
                 patient_dose_register_filename: Optional[str] = None,
                 data_split_manifest_filename: Optional[str] = None,
                 normalization: Optional[str] = None,
                 train_val_test_shuffle: Tuple[bool, bool, bool] = (True, False, False),
                 train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 crop_size: Optional[int] = None,
                 resize_size: Optional[int] = None,
                 tanh_range: bool = False,
                 data_augmentation: bool = False,
                 seed: Optional[int] = None,
                 num_workers=None):
        super().__init__()

        assert normalization is None or normalization in ["max"]
        assert task in ["autoencoder", "generation", "conditional_generation", "only_pre", "only_treatment",
                        "only_post", "only_gtv", "free_guidance_conditionnal_generation"]

        # if the task is a conditional_generation, the patient_dose_register became a mandatory file.
        if task == 'conditional_generation' or task == 'only_treatment':
            assert patient_dose_register_filename is not None

        self.__batch_size = batch_size
        self.__task = task
        self.__num_workers = num_workers if num_workers is not None else batch_size * 2
        self.__prepare_data_per_node = False
        self.__dataset_path = dataset_path
        self.__manifest_filename = manifest_filename
        self.__patient_dose_register_filename = patient_dose_register_filename
        self.__data_split_manifest_filename = data_split_manifest_filename
        self.__crop_size = crop_size
        self.__resize_size = resize_size
        self.__normalization = normalization
        self.__tanh_range = tanh_range
        self.__data_augmentation = data_augmentation
        self.__train_val_test_shuffle = train_val_test_shuffle
        self.__train_val_test_split = train_val_test_split
        # We initiate each set as an empty list. Their corresponding dataloader will be of len 0 if they stay empty.
        self.__train_set, self.__val_set, self.__test_set = None, None, None
        self.__train_patients, self.__val_patients, self.__test_patients = None, None, None
        self.__transform = None
        self.__reverse_transform = None
        self.__df_data = None
        self.__df_train_set = None
        self.__dataset = None

        if seed is not None:
            np.random.seed(seed)

    def prepare_data(self):
        pass

    def __get_patient_max_value(self, id_patient: int, images_type: str):
        patient_images = self.__df_data[self.__df_data['patient'] == id_patient]

        dataset_patient = MRI_Dataset(dataset=patient_images,
                                      dataset_path=self.__dataset_path,
                                      task=self.__task,
                                      data_augmentation=False,
                                      process_transform=self.get_process_transform(),
                                      normalization_transform=None)
        assert not patient_images.empty
        max_value = 0

        if self.__task == 'autoencoder':
            for i in range(len(patient_images)):
                if patient_images.iloc[i]['type'] == images_type:
                    mri_img = dataset_patient[i]['mri']
                    max_value = mri_img.max() if mri_img.max() > max_value else max_value
        elif self.__task in ['generation', 'free_guidance_conditionnal_generation']:
            for i in range(len(patient_images)):
                if images_type == 'preMRI':
                    img = dataset_patient[i]['preMRI']
                elif images_type == 'postMRI':
                    img = dataset_patient[i]['postMRI']
                else:
                    raise ValueError

                img_max = img.max()

                max_value = img_max if img_max > max_value else max_value
        elif self.__task == 'conditional_generation':
            for i in range(len(patient_images)):
                if images_type == 'preMRI':
                    img = dataset_patient[i]['preMRI']
                elif images_type == 'treatment':
                    img = dataset_patient[i]['treatment']
                elif images_type == 'postMRI':
                    img = dataset_patient[i]['postMRI']
                else:
                    raise ValueError

                img_max = img.max()

                max_value = img_max if img_max > max_value else max_value

        elif self.__task == 'only_pre':
            for i in range(len(patient_images)):
                pre_mri = dataset_patient[i]['preMRI']
                max_value = pre_mri.max() if pre_mri.max() > max_value else max_value
        elif self.__task == 'only_treatment':
            for i in range(len(patient_images)):
                treatment = dataset_patient[i]['treatment']
                max_value = treatment.max() if treatment.max() > max_value else max_value
        elif self.__task == 'only_post':
            for i in range(len(patient_images)):
                post_mri = dataset_patient[i]['postMRI']
                max_value = post_mri.max() if post_mri.max() > max_value else max_value
        else:
            raise NotImplementedError

        return max_value

    def __get_transform_max_norm(self, max_value_patient: float):
        assert max_value_patient != 0
        return [Lambda(lambda x: x / max_value_patient)]

    def __get_no_normalization(self):
        return [Lambda(lambda x: x)]

    def __get_normalization_transform(self, normalization: list):
        """
        contain the basic transformation structure.
        :param normalization: a list of the normalization transformations step applied.
        """
        return Compose([
            *normalization,
            Lambda(lambda x: x * 2 - 1) if self.__tanh_range else Lambda(lambda x: x)  # [0, 1] --> [-1, 1]
        ])

    def __get_rt_dose_normalization_transform(self, rt_dose_max_patient, max_dose_dataset, dose_patient):
        """
        :param rt_dose_max_patient: the max value reach in the rt dose of a patient.
        :param max_dose_dataset: the max dose present in the dataset of patient.
        :param dose_patient: the dose of the patient receive in Grey.
        """
        return Compose([
            Lambda(lambda x: x/rt_dose_max_patient),  # --> [0, 1]
            Lambda(lambda x: x * (dose_patient/max_dose_dataset)),
        ])

    def get_process_transform(self):
        """
        process_transform: common operation apply to every type of data
        """
        return Compose([
            ToTensor(),
            CenterCrop(self.__crop_size) if self.__crop_size is not None else Lambda(lambda x: x),
            Resize(self.__resize_size) if self.__resize_size is not None else Lambda(lambda x: x),
            ConvertImageDtype(torch.float32),
            Lambda(lambda x: torch.where(x < 0, torch.tensor(0, dtype=x.dtype), x))
        ])

    def save_patient_split(self, dir_path: str) -> None:
        """
        save the patient split in the given directory
        :param dir_path: the directory to save the patient split. I.E the log path of the current training.
        """
        df_split_patient = pd.DataFrame(
            {
                'patient': np.concatenate((self.__train_patients, self.__val_patients, self.__test_patients)),
                'split': ['train'] * len(self.__train_patients) + ['val'] * len(self.__val_patients) + ['test'] * len(self.__test_patients)
            }
        )
        df_split_patient.to_csv(os.path.join(dir_path, 'patient_split.csv'), index=False)

    def save_patient_split_from_dataset(self, dir_path: str):
        """
        save the dataset patient split in the given directory
        :param dir_path: the directory to save the patient split.
        """
        rows = {}

        for i in range(len(self.__train_set)):
            rows[i] = pd.Series(dict(self.__train_set.get_dataset_row(i), split='train'))
        for i in range(len(self.__val_set)):
            rows[len(self.__train_set) + i] = pd.Series(dict(self.__val_set.get_dataset_row(i), split='val'))
        for i in range(len(self.__test_set)):
            rows[len(self.__train_set) + len(self.__val_set) + i] = pd.Series(dict(self.__test_set.get_dataset_row(i), split='test'))

        pd.DataFrame(rows).T.to_csv(os.path.join(dir_path, 'patient_split_dataset.csv'), index=False)

    def setup(self, stage: str):
        """
        load dataset
        apply transform
        split dataset
        """
        self.__df_data = pd.read_csv(os.path.join(self.__dataset_path, self.__manifest_filename))

        # retrieves the list of all patient contain in our dataset.
        patients = self.__df_data['patient'].unique()

        """
        split dataset
        """
        if self.__data_split_manifest_filename is not None:
            print(f"loading data split from {self.__data_split_manifest_filename}")
            df_data_split = pd.read_csv(self.__data_split_manifest_filename)
            # we assert that all of our patient of our current dataset are present in data_split_manifest
            # if they aren't we can't use this manifest to reproduce the same split between train / val / test.
            assert sum([patient not in df_data_split['patient'].values for patient in patients]) < 1
            self.__train_patients = df_data_split[df_data_split['split'] == 'train']['patient'].values
            self.__val_patients = df_data_split[df_data_split['split'] == 'val']['patient'].values
            self.__test_patients = df_data_split[df_data_split['split'] == 'test']['patient'].values
        else:
            p_train, p_val, p_test = self.__train_val_test_split
            assert round(p_train + p_val + p_test, 2) == 1.0

            """
            compute mask to filter dataset by training, val and test by patient
            """
            indices_split = np.random.rand(len(patients))
            train_mask = indices_split < p_train
            val_mask = (p_train < indices_split) & (indices_split < p_train + p_val)
            test_mask = (p_train + p_val < indices_split) & (indices_split < p_train + p_val + p_test)

            self.__train_patients = patients[train_mask]
            self.__val_patients = patients[val_mask]
            self.__test_patients = patients[test_mask]

        self.__df_train_set = self.__df_data[self.__df_data['patient'].isin(self.__train_patients)]

        self.__dataset = MRI_Dataset(dataset=self.__df_data,
                                     dataset_path=self.__dataset_path,
                                     task=self.__task,
                                     data_augmentation=False,
                                     process_transform=self.get_process_transform(),
                                     normalization_transform=None)

        self.__train_set = MRI_Dataset(dataset=self.__df_train_set,
                                       dataset_path=self.__dataset_path,
                                       task=self.__task,
                                       data_augmentation=False,
                                       process_transform=self.get_process_transform(),
                                       normalization_transform=None)

        """
        build normalization transform for each type of image and for each patient.
        """
        self.normalization_transform = {}

        if self.__task in ["autoencoder", "generation", "free_guidance_conditionnal_generation"]:
            images_types = ['preMRI', 'postMRI']
        elif self.__task == "conditional_generation":
            images_types = ['preMRI', 'treatment', 'postMRI']
        elif self.__task == "only_pre":
            images_types = ['preMRI']
        elif self.__task == "only_treatment":
            images_types = ['treatment']
        elif self.__task == "only_post":
            images_types = ['postMRI']
        elif self.__task == "only_gtv":
            images_types = []
        else:
            raise NotImplementedError

        threshold_images_types = {}

        if 'treatment' in images_types:  # if we are handling an RT_Dose file
            df_patient_dose_register = pd.read_csv(os.path.join(self.__dataset_path, self.__patient_dose_register_filename))
            list_patient_dose = df_patient_dose_register['patient'].unique()

            # we assert that all patients of our dataset are in our patient_dose_registrer
            # likewise we assert taht all patients of our patient_dose_register are in our datasets
            assert (
                    all([patient in patients for patient in list_patient_dose])
                    and
                    all([patient in list_patient_dose for patient in patients])
            )

        for images_type in images_types:
            self.normalization_transform[str(images_type)] = {}
            for patient in patients:
                if images_type == 'treatment':
                    max_value_rt_dose = self.__get_patient_max_value(patient, 'treatment')
                    dose_patient = df_patient_dose_register[df_patient_dose_register['patient'] == patient]['dose'].values[0]
                    max_dose_dataset = df_patient_dose_register['dose'].max()

                    self.normalization_transform[str(images_type)][str(patient)] = self.__get_rt_dose_normalization_transform(
                        rt_dose_max_patient=max_value_rt_dose,
                        max_dose_dataset=max_dose_dataset,
                        dose_patient=dose_patient
                    )
                if self.__normalization == 'max':
                    if (('type' in self.__df_data.keys()) and
                            (len(self.__df_data[(self.__df_data['patient'] == patient) & (self.__df_data['type'] == images_type)]) == 0)):
                        continue
                    patient_max_value = self.__get_patient_max_value(patient, images_type=images_type)
                    self.normalization_transform[str(images_type)][str(patient)] = self.__get_normalization_transform(
                        self.__get_transform_max_norm(patient_max_value),
                    )
                elif self.__normalization is None:
                    self.normalization_transform[str(images_type)][str(patient)] = self.__get_normalization_transform(
                        self.__get_no_normalization()
                    )
                else:
                    raise NotImplementedError

        self.__train_set = MRI_Dataset(dataset=self.__df_data[self.__df_data['patient'].isin(self.__train_patients)],
                                       dataset_path=self.__dataset_path,
                                       task=self.__task,
                                       data_augmentation=self.__data_augmentation,
                                       process_transform=self.get_process_transform(),
                                       normalization_transform=self.normalization_transform)

        self.__val_set = MRI_Dataset(dataset=self.__df_data[self.__df_data['patient'].isin(self.__val_patients)],
                                     dataset_path=self.__dataset_path,
                                     task=self.__task,
                                     data_augmentation=self.__data_augmentation,
                                     process_transform=self.get_process_transform(),
                                     normalization_transform=self.normalization_transform)

        self.__test_set = MRI_Dataset(dataset=self.__df_data[self.__df_data['patient'].isin(self.__test_patients)],
                                      dataset_path=self.__dataset_path,
                                      task=self.__task,
                                      data_augmentation=self.__data_augmentation,
                                      process_transform=self.get_process_transform(),
                                      normalization_transform=self.normalization_transform)

    def get_image_normalization_transform_type_patient(self, images_type: Optional[str] = None, patient: Optional[str] = None):
        """
        return the normalization transformation object apply to a combination of images_type and patient.
        It's used in inference mode when we want to test our model on specific images that we load.
        :param images_type: the type of images (pre, post, etc.)
        :param patient: the id of the patient
        """
        if images_type is None and patient is None:
            return self.normalization_transform
        return self.normalization_transform[images_type][patient]

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=self.__train_set,
            batch_size=self.__batch_size,
            shuffle=self.__train_val_test_shuffle[0],
            # num_workers=self.__num_workers,
            # persistent_workers=True,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.__val_set,
            batch_size=self.__batch_size,
            shuffle=self.__train_val_test_shuffle[1],
            # num_workers=self.__num_workers,
            # persistent_workers=True,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.__test_set,
            batch_size=self.__batch_size,
            shuffle=self.__train_val_test_shuffle[2],
            # num_workers=self.__num_workers,
            # persistent_workers=True,
            pin_memory=True)
