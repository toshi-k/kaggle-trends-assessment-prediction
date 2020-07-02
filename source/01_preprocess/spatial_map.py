from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import nilearn as nl
from tqdm import tqdm

import h5py


def load_subject(mat_filename: str, mask_ni_img: nl.nifti1.Nifti1Image):
    """
    This function is forked from Rohit's kaggle notebook.
    TReNDS - EDA + Visualization + Simple Baseline
    https://www.kaggle.com/rohitsingh9990/trends-eda-visualization-simple-baseline

    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.

    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers

    """
    with h5py.File(str(mat_filename), 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    subject_ni_img = nl.image.new_img_like(mask_ni_img, subject_data,
                                           affine=mask_ni_img.affine, copy_header=True)

    arr = subject_ni_img.dataobj.transpose(3, 0, 1, 2) * 3000

    # max_value = np.max(arr)
    # min_value = np.min(arr)
    # print(f'format: max_value {max_value} min_value {min_value}')

    return arr.astype(np.int16)


def load_ids(dir_dataset: Path):

    train_scores = pd.read_csv(dir_dataset / 'train_scores.csv')
    sample_submission = pd.read_csv(dir_dataset / 'sample_submission.csv')

    train_ids = train_scores['Id'].tolist()
    test_ids = sample_submission['Id'].tolist()
    test_ids = sorted(list(set([int(s[:5]) for s in test_ids])))

    return train_ids, test_ids


def save_spatial_map(dir_dataset: Path,
                     ids: List[int],
                     mask_ni_img: nl.nifti1.Nifti1Image,
                     dataset_type: str):

    spatial_map = np.zeros((len(ids), 53, 53, 63, 52), dtype=np.int16)

    for i, _id in enumerate(tqdm(ids)):
        spatial_map[i] = load_subject(str(dir_dataset / f'fMRI_{dataset_type}' / f'{_id}.mat'), mask_ni_img)

    np.save(f'../../input/spatial_map_{dataset_type}.npy', spatial_map)


def main():

    dir_dataset = Path('../../dataset')

    train_ids, test_ids = load_ids(dir_dataset)

    mask_ni_img = nl.image.load_img(str(dir_dataset / 'fMRI_mask.nii'))

    save_spatial_map(dir_dataset, train_ids, mask_ni_img, 'train')
    save_spatial_map(dir_dataset, test_ids, mask_ni_img, 'test')


if __name__ == '__main__':
    main()
