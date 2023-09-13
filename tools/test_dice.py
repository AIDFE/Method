import torch
import numpy as np
from torch.nn import functional as F
from tools.util import DiceScore


def prediction_wrapper(model, test_loader, opt, epoch, label_name, mode = 'base', save_prediction = False):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        recomp_img_list = []
        for idx, batch in enumerate(test_loader):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['lb'].shape[0] == 1 # enforce a batchsize of 1

            gth = batch['lb'].cuda()
            pred = model(batch['img'].cuda())
            pred = torch.argmax(pred, 1)
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['img'][0, 1,...].numpy()
            slice_idx += 1
            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name ), model, label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names


def eval_list_wrapper(vol_list, nclass, model, label_name):
    """
    Evaluatation and arrange predictions
    """
    ScoreDiceEval = DiceScore(nclass, ignore_chan0 = False).cuda() 

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    conf_mat_list = [] # confusion matrices
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],
                    'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices = ScoreDiceEval(torch.unsqueeze(pred_, 1), gth_, dense_input = True).cpu().numpy() # this includes the background class
        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f} \n, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc

    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)

    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain

    return error_dict, dsc_table, domain_names
    