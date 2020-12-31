# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import glob
import json
from PIL import Image
import numpy as np
import cv2
import warnings

from fblib.util.mypath import Path


def sync_and_evaluate_one_folder(database, save_dir, exp_name, prefix=None, all_tasks_present=False):
    # dataset specific parameters
    if database == 'BSDS500':
        num_req_files = 200
        gt_set = ''
    elif database == 'PASCALContext':
        if all_tasks_present:
            num_req_files = 1853
            gt_set = 'val_all_tasks_present'
        else:
            num_req_files = 5105
            gt_set = 'val'
    elif database == 'NYUD':
        num_req_files = 654
        gt_set = 'val'
    else:
        raise NotImplementedError

    if prefix is None:
        res_exp_name = exp_name
    else:
        res_exp_name = prefix + '_' + exp_name

    # Check whether results of experiments exist
    chk_dir = os.path.join(save_dir, exp_name, 'Results_' + database, 'edge')
    if not os.path.exists(chk_dir):
        print('Experiment {} is not yet ready. Omitting this directory'.format(exp_name))
        return

    # Check for filenames
    fnames = sorted(glob.glob(os.path.join(chk_dir, '*')))
    if len(fnames) < num_req_files:
        print('Something is wrong with this directory. Check required: {}'.format(exp_name))
        return
    elif len(fnames) > num_req_files:
        print('Already synced: {}'.format(exp_name))
    else:
        # Seism path
        seism_cluster_dir = Path.seism_root_dir()

        # rsync to seism
        rsync_str = 'rsync -aP {}/ '.format(chk_dir)
        rsync_str += 'kmaninis@reinhold.ee.ethz.ch:{}/datasets/{}/{} '.format(seism_cluster_dir, database, res_exp_name)
        rsync_str += '--exclude=models --exclude=*.txt'
        print(rsync_str)
        os.system(rsync_str)

        # Submit the job
        subm_job_str = 'ssh kmaninis@reinhold.ee.ethz.ch  "source /home/sgeadmin/BIWICELL/common/settings.sh;' \
                       'source /home/sgeadmin/BIWICELL/common/settings.sh;'
        subm_job_str += 'cp {}/parameters/HED.txt {}/parameters/{}.txt; ' \
                        ''.format(seism_cluster_dir, seism_cluster_dir, res_exp_name)
        subm_job_str += 'qsub -N evalFb -t 1-102 {}/eval_in_cluster.py {} read_one_cont_png fb 1 102 {} {}"' \
                        ''.format(seism_cluster_dir, res_exp_name, database, gt_set)
        print(subm_job_str)
        os.system(subm_job_str)

        # Leave the proof of submission
        os.system('touch {}/SYNCED_TO_REINHOLD'.format(chk_dir))


def sync_evaluated_results(database, save_dir, exp_name, prefix=None):
    if prefix is not None:
        res_exp_name = prefix + '_' + exp_name
    else:
        res_exp_name = exp_name

    split = 'val'

    # Check whether results of experiment exists
    chk_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    if not os.path.exists(chk_dir):
        print('Experiment {} is not yet ready. Omitting this directory'.format(exp_name))
        return

    chk_file = os.path.join(save_dir, exp_name, 'Results_' + database,
                            database + '_' + split + '_' + exp_name + '_edge.json')

    if os.path.isfile(chk_file):
        with open(chk_file, 'r') as f:
            eval_results = json.load(f)
    else:
        print('Creating json: {}'.format(res_exp_name))
        eval_results = {}
        for measure in {'ods_f', 'ois_f', 'ap'}:
            tmp_fname = os.path.join(Path.seism_root_dir(), 'results', 'pr_curves', database,
                                     database + '_' + split + '_fb_' + res_exp_name + '_' + measure + '.txt')
            if not os.path.isfile(tmp_fname):
                print('Result not available')
                continue

            with open(tmp_fname, 'r') as f:
                eval_results[measure] = float(f.read().strip())

        # Create edge json file
        if eval_results:
            print('Saving into .json: {}'.format(chk_file))
            with open(chk_file, 'w') as f:
                json.dump(eval_results, f)

    for measure in eval_results:
        print('{}: {}'.format(measure, eval_results[measure]))


def sync_and_evaluate_subfolders(p, database):
    print('Starting check in parent directory: {}'.format(p['save_dir_root']))
    dirs = os.listdir(p['save_dir_root'])
    for exp in dirs:
        sync_and_evaluate_one_folder(database=database,
                                     save_dir=p['save_dir_root'],
                                     exp_name=exp,
                                     prefix=p['save_dir_root'].split('/')[-1],
                                     all_tasks_present=(exp.find('mini') >= 0))


def gather_results(p, database):
    print('Gathering results: {}'.format(p['save_dir_root']))
    dirs = os.listdir(p['save_dir_root'])
    for exp in dirs:
        sync_evaluated_results(database=database,
                               save_dir=p['save_dir_root'],
                               exp_name=exp,
                               prefix=p['save_dir_root'].split('/')[-1])


def eval_and_store_edges(database, save_dir, exp_name, overfit):

    # unofficial evaluation of edges; NO SAVING, no overfit implementation
    # just rmse and mean err

    if overfit:
        raise NotImplementedError

    # Dataloaders
    if database == 'NYUD':
        from fblib.dataloaders import nyud as nyud
        gt_set = 'val'
        db = nyud.NYUD_MT(split=gt_set, do_edge=True, overfit=overfit)
    elif database == 'PASCALContext':
        from fblib.dataloaders import pascal_context as pascal_context
        gt_set = 'val'
        db = pascal_context.PASCALContext(split=gt_set, do_edge=True, overfit=overfit)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    eval_results = eval_edges(db, os.path.join(res_dir, 'edge'))

    print('Results for Depth Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))







def eval_edges(loader, folder):
    # unofficial evaluation of edges; no saving, no overfit implementation
    # just rmse and mean err

    rmses = []
    log_rmses = []

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating edges: {} of {} objects'.format(i, len(loader)))

        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        pred = np.asarray(Image.open(filename)).astype(np.float32) / 255

        label = sample['edge']

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_LINEAR)

        valid_mask = (label < 1e-9)
        pred = np.ma.array(pred, valid_mask).filled(0)
        label = np.ma.array(label, valid_mask).filled(0)  # set values to 0, wherever the label value is smaller than 1e-9

        n_valid = np.sum(valid_mask)                      # only evaluate, where we would find edges

        log_rmse_tmp = (np.log(label) - np.log(pred)) ** 2
        log_rmse_tmp = np.sqrt(np.sum(log_rmse_tmp) / n_valid)
        log_rmses.extend([log_rmse_tmp])

        rmse_tmp = (label - pred) ** 2
        rmse_tmp = np.sqrt(np.sum(rmse_tmp) / n_valid)
        rmses.extend([rmse_tmp])

    rmses = np.array(rmses)
    log_rmses = np.array(log_rmses)

    eval_result = dict()
    eval_result['rmse'] = np.mean(rmses)
    eval_result['log_rmse'] = np.median(log_rmses)

    return eval_result



def main():
    exp_root_dir = os.path.join(Path.exp_dir(), 'pascal_se')
    edge_dirs = glob.glob(os.path.join(exp_root_dir, 'edge*'))

    p = {}

    for edge_dir in edge_dirs:
        p['save_dir_root'] = os.path.join(exp_root_dir, edge_dir)
        # sync_and_evaluate_subfolders(p, 'NYUD')
        gather_results(p, 'PASCALContext')


if __name__ == '__main__':
    main()
