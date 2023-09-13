import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import itertools
import glob
import shutil
import time
import numpy as np
import torch
from tools.util import AttrDict, worker_init_fn, SoftDiceLoss
from torch.utils.data import DataLoader
from tools.vis import dataset_vis
from tools.test_dice import prediction_wrapper
from networks.deeplabv3p import DeepLabv3p
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import torch.nn as nn

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
    name = "deeplabv3-seg"  # exp_name
    checkpoints_dir = './checkpoints'
    snapshot_dir = ''
    epoch_count = 2000
    batchsize = 20
    infer_epoch_freq = 50
    save_epoch_freq = 50
    lr = 0.0003
    
    data_name = 'ABDOMINAL'
    tr_domain = 'CT'
    te_domain = 'MR'

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
        val_set         = ABD.get_test(modality = opt.tr_domain, norm_func = train_set.normalize_op)
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
    model.train()

    criterionDice = SoftDiceLoss(opt.num_classes).cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.00003)
    tbfile_dir = os.path.join(_run.observers[0].dir, 'tboard_file'); os.mkdir(tbfile_dir)
    writer = SummaryWriter(tbfile_dir)

    iter_num = 0
    max_iterations = opt.epoch_count * len(train_loader)
    best_score = 0
   
    for epoch in range(opt.epoch_count):
        epoch_start_time = time.time()
        for i, train_batch in enumerate(train_loader):
            img = train_batch['img'].cuda()
            lb = train_batch['lb'].cuda()
            outputs = model(img)
            loss_dice = criterionDice(outputs, lb)
            loss = loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = opt.lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # _log.info('iteration %d / %d : loss : %f, loss_ce: %f' % (iter_num, max_iterations, loss.item(), loss_ce.item()))

            if iter_num % 200 == 0:
                image = img[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = lb[1,0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            

        # test
        if (epoch % opt.infer_epoch_freq == 0):
            model.eval()
            t0  = time.time()
            print('infering the model at the end of epoch %d' % (epoch))
            with torch.no_grad():
                print(f'Starting inferring ... ')
                preds, dsc_table, error_dict, domain_list = prediction_wrapper(model, test_loader, opt, epoch, label_name)
                preds_val, dsc_table_val, error_dict_val, domain_list_val = prediction_wrapper(model, val_loader, opt, epoch, label_name)

                if len(opt.te_domain) == 1:
                    if best_score < error_dict['overall']:
                        if best_score != 0:
                            model_path = os.listdir(opt.snapshot_dir)[0]
                            os.remove(os.path.join(opt.snapshot_dir, model_path))
                        best_score = error_dict['overall']
                        save_mode_path = os.path.join(opt.snapshot_dir, f'best_score_{best_score*100:.2f}_{epoch}' + '.pth')
                        torch.save(model.state_dict(), save_mode_path)
                    _run.log_scalar('meanDiceTarget', error_dict['overall'])
                    _run.log_scalar('val_meanDiceTarget', error_dict_val['overall'])

                else:
                    if best_score < error_dict['overall_by_domain']:
                        if best_score != 0:
                            model_path = os.listdir(opt.snapshot_dir)[0]
                            os.remove(os.path.join(opt.snapshot_dir, model_path))
                        best_score = error_dict['overall_by_domain']
                        save_mode_path = os.path.join(opt.snapshot_dir, f'best_score_{best_score*100:.2f}_{epoch}' + '.pth')
                        torch.save(model.state_dict(), save_mode_path)
                    _run.log_scalar('meanDiceAvgTargetDomains', error_dict['overall_by_domain'])
                    _run.log_scalar('val_meanDiceAvgTargetDomains', error_dict_val['overall'])
                    for _dm in domain_list_val:
                        _run.log_scalar(f'val_meanDice_{_dm}', error_dict_val[f'domain_{_dm}_overall'])
                    for _dm in domain_list:
                        _run.log_scalar(f'meanDice_{_dm}', error_dict[f'domain_{_dm}_overall'])

                t1 = time.time()
                print("End of model inference, which takes {} seconds".format(t1 - t0))
            model.train()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.epoch_count, time.time() - epoch_start_time))
            

            
          
            
        











    