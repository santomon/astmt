# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import socket
import timeit
import cv2
from datetime import datetime
import imageio
import numpy as np

import sys
sys.path.append("./../../../")

# PyTorch includes
import torch
import torch.optim as optim
from torch.nn.functional import interpolate

# Custom includes
from fblib.util.helpers import generate_param_report
from fblib.util.dense_predict.utils import lr_poly
from experiments.dense_predict import common_configs
from fblib.util.mtl_tools.multitask_visualizer import TBVisualizer, visualize_network
from fblib.util.model_resources.flops import compute_gflops
from fblib.util.model_resources.num_parameters import count_parameters
from fblib.util.dense_predict.utils import AverageMeter

# Custom optimizer
from fblib.util.optimizer_mtl.select_used_modules import make_closure

# Configuration file
from experiments.dense_predict.pascal_resnet import config as config

# Tensorboard include
from tensorboardX import SummaryWriter


def main():
    p = config.create_config()

    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    p.TEST.BATCH_SIZE = 32

    # Setting parameters
    n_epochs = p['epochs']

    print("Total training epochs: {}".format(n_epochs))
    print(p)
    print('Training on {}'.format(p['train_db_name']))

    snapshot = 10  # Store a model every snapshot epochs
    test_interval = p.TEST.TEST_INTER  # Run on test set every test_interval epochs
    torch.manual_seed(p.SEED)

    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    if not os.path.exists(os.path.join(p['save_dir'], 'models')):
        if p['resume_epoch'] == 0:
            os.makedirs(os.path.join(p['save_dir'], 'models'))
        else:
            if not config.check_downloaded(p):
                print('Folder does not exist.No checkpoint to resume from. Exiting')
                exit(1)

    net = config.get_net_resnet(p)

    # Visualize the network
    if p.NETWORK.VIS_NET:
        visualize_network(net, p)

    gflops = compute_gflops(net, in_shape=(p['trBatch'], 3, p.TRAIN.SCALE[0], p.TRAIN.SCALE[1]),
                            tasks=p.TASKS.NAMES[0])
    print('GFLOPS per task: {}'.format(gflops / p['trBatch']))

    print('\nNumber of parameters (in millions): {0:.3f}'.format(count_parameters(net) / 1e6))
    print('Number of parameters (in millions) for decoder: {0:.3f}\n'.format(count_parameters(net.decoder) / 1e6))

    net.to(device)





    # Generate Results
    net.eval()
    _, _, transforms_infer = config.get_transformations(p)
    for db_name in p['infer_db_names']:

        testloader = config.get_test_loader(p, db_name=db_name, transforms=transforms_infer, infer=True)
        save_dir_res = os.path.join(p['save_dir'], 'Results_' + db_name)

        print('Testing Network')
        # Main Testing Loop
        with torch.no_grad():
            for ii, sample in enumerate(testloader):

                img, meta = sample['image'], sample['meta']

                # Forward pass of the mini-batch
                inputs = img.to(device)
                tasks = net.tasks

                for task in tasks:
                    output, _ = net.forward(inputs, task=task)

                    save_dir_task = os.path.join(save_dir_res, task)
                    if not os.path.exists(save_dir_task):
                        os.makedirs(save_dir_task)

                    output = interpolate(output, size=(inputs.size()[-2], inputs.size()[-1]),
                                         mode='bilinear', align_corners=False)
                    output = common_configs.get_output(output, task)

                    for jj in range(int(inputs.size()[0])):
                        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
                            continue

                        fname = meta['image'][jj]

                        result = cv2.resize(output[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
                                            interpolation=p.TASKS.INFER_FLAGVALS[task])

                        imageio.imwrite(os.path.join(save_dir_task, fname + '.png'), result.astype(np.uint8))

    if p.EVALUATE:
        common_configs.eval_all_results(p)


if __name__ == '__main__':
    main()
