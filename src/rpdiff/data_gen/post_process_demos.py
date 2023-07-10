import os, os.path as osp
import sys
import numpy as np
import shutil
import random

from rpdiff.utils import path_util


def npz2dict(npz):
    out_dict = {f: npz[f] for f in npz.files}
    return out_dict


def merge_seeds(merged_dir):

    if not osp.exists(merged_dir):
        os.makedirs(merged_dir)

    root_name = merged_dir.split('/')[-1]
    all_demos_dir = '/'.join(merged_dir.split('/')[:-1])

    print(f'\n\nRoot data directory: {root_name}\n\n')

    # datadirs = [osp.join(os.getcwd(), fn) for fn in os.listdir('.') if root_name in fn and 'seed' in fn]
    datadirs = [osp.join(all_demos_dir, fn) for fn in os.listdir(all_demos_dir) if root_name in fn and 'seed' in fn]

    print(f'\n\nLoading from datadirs: {datadirs}\n\n')

    for j, datadir in enumerate(datadirs):
        for root, dirs, files in os.walk(datadir):
            print(f'root: {root}, dirs: {dirs}, files: {files}')

            root_leaf_dir = root.split('/')[-1]
            if root_leaf_dir == 'eval' or 'trial_' in root_leaf_dir or '_imgs' in root_leaf_dir:
                print(f'Skipping...')
                continue

            if not len(files):
                if not len(dirs):
                    continue

                task_name_dir = dirs[0]
                if task_name_dir == 'eval':
                    print(f'Skipping (don"t need "eval" dir)...')
                    continue

                new_save_dir = osp.join(merged_dir, task_name_dir)
                if not osp.exists(new_save_dir):
                    os.makedirs(new_save_dir)

                continue

            for i, fn in enumerate(files):
                full_name = osp.join(root, fn)
                new_full_name = osp.join(new_save_dir, fn).replace('.npz', f'_{j}.npz')

                # print(f'\n\nOld: {full_name}\nNew: {new_full_name}\n\n')
                shutil.copy(full_name, new_full_name)

            print(f'Done copying files from {root} into {new_save_dir}')
            print('\n\n\n------------------\n\n\n')


def make_chunked_data(datadir, avg_mb_per_file=10):
    assert osp.exists(datadir)
    datadir_new = datadir + '_chunked'

    if not osp.exists(datadir_new):
        os.makedirs(datadir_new)

    # current size per file and number of files
    # avg_mb_per_file = 16
    # avg_mb_per_file = 10
    # avg_mb_per_file = 1
    total_start_files = len(os.listdir(datadir))
    total_mb = avg_mb_per_file * len(os.listdir(datadir))

    # approx desired size per file and number of files = 100mb and <1000
    n_files_per_new = int(100.0 / avg_mb_per_file) - 1 
    n_files_new_total = int(total_mb / 100.0)
    print(f'total_mb: {total_mb}')
    print(f'n_files_new_total: {n_files_new_total}')

    # get number of subfolders we will need to create
    n_folders_new_total = (n_files_new_total // 1000) + 1
    datadir_folders_new = [osp.join(datadir_new, str(folder)) for folder in range(n_folders_new_total)]

    # get all the filenames
    fnames = [osp.join(datadir, fn) for fn in os.listdir(datadir) if fn.endswith('.npz')]
    # fnames_new = [osp.join(datadir_new, f'demo_aug_{i}.npz') for i in range(n_files_new_total)]

    n_files_total = 0
    fnames_new =  []
    files_per_next_folder = min(1000, n_files_new_total)
    for i, folder in enumerate(datadir_folders_new):
        if not osp.exists(folder):
            os.makedirs(folder)
        print(f'Saving {files_per_next_folder} files to folder: {folder}')
        folder_fnames_new = [osp.join(folder, f'demo_aug_{(n_files_total + j)}.npz') for j in range(files_per_next_folder)]
        fnames_new.extend(folder_fnames_new)
        n_files_total += len(folder_fnames_new)

        # calc how many files go in the next folder, should not run if we have run out
        files_per_next_folder = min(1000, n_files_new_total - 1000*(i+1))

        if files_per_next_folder < 0:
            print('Reached end of set of files')
            break

    print('\n\nSaving to new folders\n\n')

    cur_idx = 0
    for i, fn in enumerate(fnames_new):
        if osp.exists(fn):
            chunked_data = np.load(fn, allow_pickle=True)
            from IPython import embed; embed()
            assert False

        # fetch the current files we want to chunk
        current_fnames = fnames[cur_idx:cur_idx+n_files_per_new]
        current_data = [np.load(fn, allow_pickle=True) for fn in current_fnames]

        # keys based on index in the new file
        save_keys = np.arange(n_files_per_new).tolist()
        save_data_dict = {str(save_keys[i]) : npz2dict(current_data[i]) for i in range(len(current_data))}

        # save new data
        print(f'Saving to filename: {fn}')
        np.savez(fn, **save_data_dict)

        cur_idx += n_files_per_new

    return datadir_new


def get_split_files(file_list, n_test, n_val):
    test_files = []
    train_val_files = []
    train_files = []

    for fn in file_list:
        if not fn.endswith('.npz'):
            continue

        test_files.append(fn)

        if len(test_files) >= n_test:
            break

    for fn in file_list:
        if not fn.endswith('.npz'):
            continue
        if fn in test_files:
            continue
        if len(train_val_files) >= n_val:
            break

        train_val_files.append(fn)

    train_files = []
    for fn in file_list:
        if not fn.endswith('.npz'):
            continue
        if fn in test_files:
            continue
        if fn in train_val_files:
            continue
        train_files.append(fn)

    return test_files, train_val_files, train_files


def make_dataset_splits(datadir, chunked=False):

    split_dir = osp.join(datadir, 'split_info')
    if not osp.exists(split_dir):
        os.makedirs(split_dir)

    train_split_str = ''
    train_val_split_str = ''
    test_split_str = ''

    train_split_fname = osp.join(split_dir, 'train_split.txt')
    train_val_split_fname = osp.join(split_dir, 'train_val_split.txt')
    test_split_fname = osp.join(split_dir, 'test_split.txt')
    
    if chunked:
        for folder in os.listdir(datadir):
            if 'split_info' in folder:
                continue

            sub_datadir = osp.join(datadir, folder)
            files = os.listdir(sub_datadir)
            files = sorted(files)

            n_data_per_file = 10

            # test set - 10% of data
            n_test = max(2, int(len(files) * 0.1 / n_data_per_file))
            # val set - 10% of data
            n_val = max(2, int(len(files) * 0.1 / n_data_per_file))

            test_files, train_val_files, train_files = get_split_files(files, n_test, n_val) 

            train_split_str += '\n'.join(train_files)
            train_val_split_str += '\n'.join(train_val_files)
            test_split_str += '\n'.join(test_files)
    else:
        files = [fn for fn in os.listdir(datadir) if 'split_info' not in fn]
        files = sorted(files)

        # test set - 10% of data
        n_test = max(2, int(len(files) * 0.1))
        # val set - 10% of data
        n_val = max(2, int(len(files) * 0.1))

        test_files, train_val_files, train_files = get_split_files(files, n_test, n_val) 

        train_split_str += '\n'.join(train_files)
        train_val_split_str += '\n'.join(train_val_files)
        test_split_str += '\n'.join(test_files)

    with open(train_split_fname, 'w') as f:
        f.write(train_split_str)
    with open(train_val_split_fname, 'w') as f:
        f.write(train_val_split_str)
    with open(test_split_fname, 'w') as f:
        f.write(test_split_str)

    return split_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_dir', type=str, default='data/task_demos', help='Root directory where the demos are saved (so we can run this script from elsewhere in the repo)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Name of the folder to create + post-process. This should be the "root" name in the folders that were saved after running data gen (i.e., without "_seedN" attached, the "experiment_name" used in the config file when running data gen)')
    parser.add_argument('--no_merge', action='store_true')
    parser.add_argument('--no_chunk', action='store_true')

    args = parser.parse_args()
    
    dataset_dir = osp.join(path_util.get_rpdiff_src(), args.root_data_dir, args.dataset_dir)
    print(f'Root dataset dir: {dataset_dir} (directory may not exist yet, until merging complete)')
    
    if args.no_merge:
        pass
    else:
        merge_seeds(dataset_dir)

    task_name_datadir = [fn for fn in os.listdir(dataset_dir) if 'task_name' in fn and 'chunked' not in fn][0]
    task_name_datadir_full = osp.join(dataset_dir, task_name_datadir)
    print(f'Directory inside root containing the actual files: {task_name_datadir_full}')
    
    if args.no_chunk:
        chunked_task_name_datadir_full = task_name_datadir_full + '_chunked'
    else:
        chunked_task_name_datadir_full = make_chunked_data(task_name_datadir_full)
    print(f'Chunked data directory name: {chunked_task_name_datadir_full}')

    out_split_dir = make_dataset_splits(task_name_datadir_full)
    chunked_out_split_dir = make_dataset_splits(chunked_task_name_datadir_full, chunked=True)

    print(f'Dataset split files in folder: {out_split_dir}')
    print(f'Chunked dataset split files in folder: {chunked_out_split_dir}')
