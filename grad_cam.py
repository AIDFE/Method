from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import itertools
import glob
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from tools.util import AttrDict, worker_init_fn, SoftDiceLoss
from torch.utils.data import DataLoader
from tools.vis import dataset_vis, to01
from tools.test_dice import prediction_wrapper
from networks.deeplabv3p import DeepLabv3p
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tensorboardX import SummaryWriter

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('MedAI') 
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './my_utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))

for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config          # Sacred 相关
def cfg():

    seed = 1234
    name = "grad_cam"  # exp_name
    checkpoints_dir = './checkpoints'
    snapshot_dir = ''
    epoch_count = 1
    batchsize = 1
    infer_epoch_freq = 50
    save_epoch_freq = 50
    lr = 0.0003
    
    data_name = 'ABDOMINAL'
    tr_domain = 'MR'
    te_domain = 'CT'

    # data_name = 'MMS'
    # tr_domain = ['vendorA']
    # te_domain = ['vendorC']

    num_classes = 5
    patch_size = 16
    in_channels = 3


@ex.config_hook     # Sacred 相关
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{config["name"]}'
    observer = FileStorageObserver.create(os.path.join(config['checkpoints_dir'], exp_name))
    ex.observers.append(observer)
    return config


@ex.automain
def main(_run, _config, _log):
    # configs for sacred
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'), exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        _config['run_dir'] = _run.observers[0].dir
        _config['snapshot_dir'] = f'{_run.observers[0].dir}/snapshots'

    opt = AttrDict(_config)

     # load dataset
    if opt.data_name == 'ABDOMINAL':
        import dataloaders.abd_dataset as ABD
        if not isinstance(opt.tr_domain, list):
            opt.tr_domain = [opt.tr_domain]
            opt.te_domain = [opt.te_domain]

        train_set       = ABD.get_training(modality = opt.tr_domain)
        val_set         = ABD.get_validation(modality = opt.tr_domain, norm_func = train_set.normalize_op)
        if opt.te_domain[0] == opt.tr_domain[0]:
            test_set        = ABD.get_test(modality = opt.te_domain, norm_func = train_set.normalize_op) 
        else:
            test_set        = ABD.get_test_all(modality = opt.te_domain, norm_func = None)
        label_name          = ABD.LABEL_NAME
    
    elif opt.data_name == 'MMS':
        import dataloaders.mms_dataset as MMS
        if not isinstance(opt.tr_domain, list):
            opt.tr_domain = [opt.tr_domain]
            opt.te_domain = [opt.te_domain]

        train_set       = MMS.get_training(modality = opt.tr_domain)
        val_set         = MMS.get_validation(modality = opt.tr_domain, norm_func = train_set.normalize_op)
        if opt.te_domain[0] == opt.tr_domain[0]:
            test_set        = MMS.get_test(modality = opt.te_domain, norm_func = train_set.normalize_op) 
        else:
            test_set        = MMS.get_test_all(modality = opt.te_domain, norm_func = None)
        label_name          = MMS.LABEL_NAME
    
    else:
        raise NotImplementedError 
    

    _log.info(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    # dataset_vis(test_set, save_path='CT', vis_num=10)     # dataset 可视化验证：数据和标签

    train_loader = DataLoader(dataset = train_set, num_workers = 4,\
                              batch_size = opt.batchsize, shuffle = True, 
                              drop_last = True, worker_init_fn =  worker_init_fn, 
                              pin_memory = True)
    
    val_loader = DataLoader(dataset = val_set, num_workers = 4,\
                             batch_size = 1, shuffle = False, pin_memory = True)
    
    
    test_loader = DataLoader(dataset = test_set, num_workers = 4,\
                             batch_size = 1, shuffle = False, pin_memory = True)
    
    model = DeepLabv3p(opt, num_classes=opt.num_classes, norm_layer=nn.BatchNorm2d, 
                       in_channels=opt.in_channels,
                       pretrained_model='pretrain_model/resnet50_v1c.pth').cuda()
    
    save_dict = torch.load('/home/lijiaxi/Project/MedAI/checkpoints/deeplabv3-seg/1/snapshots/best_score_76.63_1100.pth', map_location='cuda:0')
    model.load_state_dict(save_dict, strict=False)
    
    cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1] ], use_cuda=True)
    # cam = GradCAM(model=model, target_layers=[model.head.last_conv], use_cuda=True)
    # cam = GradCAM(model=model, target_layers=[model.head.aspp.leak_relu], use_cuda=True)


    cam_save_path = 'visualization/grad_cam_deep_feature_MR_2'
    isExists=os.path.exists(cam_save_path)
    if not isExists:
        os.mkdir(cam_save_path) 

    for epoch in range(opt.epoch_count):
        for i, train_batch in enumerate(train_loader):
            img = train_batch['img'].cuda()
            lb = train_batch['lb'].cuda()
            targets = [SemanticSegmentationTarget(2, lb)]
            epoch_start_time = time.time()
            grayscale_cam = cam(input_tensor=img, targets=targets) 
            epoch_end_time = time.time()

            print(epoch_end_time - epoch_start_time)
            grayscale_cam = grayscale_cam[0,:]
            img_float_np = np.float32(to01(img[0,0:1,...]).permute((1, 2, 0)).cpu().numpy())/255
            cam_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=False)
            cv2.imwrite(os.path.join(cam_save_path, train_batch['scan_id'][0] + '_' + str(train_batch['z_id'][0].numpy()) + '.png'), cam_image)

            if i == 30:
                break

          

            
          
            
        





