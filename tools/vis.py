import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

colors = np.array([
  [0,     0,   0],
  [145, 193,  62],
  [254, 232,  81], 
  [ 29, 162, 220], 
  [238,  37,  36]]) 



def to01(x, by_channel = False):
  if not by_channel:
      out = (x - x.min()) / (x.max() - x.min()) * 255
  else:
      nb, nc, nh, nw = x.shape
      xmin = x.view(nb, nc, -1).min(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
      xmax = x.view(nb, nc, -1).max(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
      out = (x - xmin + 1e-5) / (xmax - xmin + 1e-5)
  return out



def overlay_seg_img(img, seg):
  # get unique labels
  labels = np.unique(seg)
  # remove background
  labels = labels[labels != 0]
  # img backgournd
  img_b = img*(seg == 0)
  # final_image
  final_img = np.zeros([img.shape[0], img.shape[1], 3])
  final_img += img_b[:, :, np.newaxis]

  for l in labels:
      mask = seg == l
      img_f = img*mask
      # convert to rgb
      img_f = np.tile(img_f, (3, 1, 1)).transpose(1, 2, 0)
      # colored segmentation
      img_seg = colors[l*mask]
      # alpha overlay
      final_img += 0.5*img_f + 0.5*img_seg
  
  return final_img


def dataset_vis(dataset, save_path, vis_num):
    vis_loader = DataLoader(dataset = dataset, num_workers = 1,\
                            batch_size = 1, shuffle = True, pin_memory = True)

    for idx, batch in enumerate(vis_loader):
        if idx == vis_num:
           break
        vis_input = {
                      'img': batch['img'],
                      'lb': batch['lb']
                     }
        seg = vis_input['lb'].long().sum(dim=1).cpu().numpy()
        final_image = overlay_seg_img(to01(vis_input['img'])[0,0].cpu().numpy(), seg[0])
        isExists=os.path.exists(save_path)
        if not isExists:
          os.mkdir(save_path) 
        Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(save_path, batch['scan_id'][0] + '_' + str(batch['z_id'][0].numpy()) + '.png'))

   

#  print(masks_enc[0].shape, masks_pred[0].shape) #[B, 12, 12]
#             print(len(masks_enc), len(masks_pred))

#             re_mask_enc = F.interpolate(masks_enc[0].unsqueeze(1).float(), size=[192, 192], mode="nearest")
#             print(re_mask_enc.shape)

#             img_mask_enc = img * re_mask_enc
#             final_image = to01(img_mask_enc)[0,0].cpu().numpy()
#             Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(save_path, train_batch['scan_id'][0] + '_' + str(train_batch['z_id'][0].numpy()) + '.png'))

            
#             re_mask_p_1 = F.interpolate(masks_pred[0].unsqueeze(1).float(), size=[192, 192], mode="nearest")
#             re_mask_p_2 = F.interpolate(masks_pred[1].unsqueeze(1).float(), size=[192, 192], mode="nearest")
#             re_mask_p_3 = F.interpolate(masks_pred[2].unsqueeze(1).float(), size=[192, 192], mode="nearest")
#             re_mask_p_4 = F.interpolate(masks_pred[3].unsqueeze(1).float(), size=[192, 192], mode="nearest")


#             img_mask_p1 = img * re_mask_p_1
#             final_image = to01(img_mask_p1)[0,0].cpu().numpy()
#             print(final_image)
#             Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(save_path_p, train_batch['scan_id'][0] + '_' + str(train_batch['z_id'][0].numpy()) + '1.png'))

#             img_mask_p2 = img * re_mask_p_2
#             final_image = to01(img_mask_p2)[0,0].cpu().numpy()
#             Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(save_path_p, train_batch['scan_id'][0] + '_' + str(train_batch['z_id'][0].numpy()) + '2.png'))

#             img_mask_p3 = img * re_mask_p_3
#             final_image = to01(img_mask_p3)[0,0].cpu().numpy()
#             Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(save_path_p, train_batch['scan_id'][0] + '_' + str(train_batch['z_id'][0].numpy()) + '3.png'))

#             img_mask_p4 = img * re_mask_p_4
#             final_image = to01(img_mask_p4)[0,0].cpu().numpy()
#             Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(save_path_p, train_batch['scan_id'][0] + '_' + str(train_batch['z_id'][0].numpy()) + '4.png'))


#             if i == 10:
#                 break
            


