import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import os

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def recursive_init(cls, myDict):
        """
        recursively look into a dictionary and convert each sub_dictionary entry to AttrDict
        This is a little bit messy
        """
#        myDict = copy.deepcopy(myDict)

        def _rec_into_subdict(curr_dict):
            for key, entry in curr_dict.items():
                if type(entry) is dict:
                    _rec_into_subdict(entry)
                    curr_dict[key] = cls(entry)

        _rec_into_subdict(myDict)
        return cls(myDict)


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()

    def forward(self, X_in):
        """
        Args:
            added
        """
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth]) #[nb/nx/nz, ...., nc]
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        if n_dim > 1:
            return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float() # [nb/nx/n\, nc, ny]
        else:
            return out.float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class DiceScore(nn.Module):
    """ WARNING: This is not a loss function! don't use this. To train a network, use ``SoftDiceLoss'' above.
        Improving memory efficiency
    """

    def __init__(self, n_classes, ignore_chan0 = True):
        """
        Args:
            ignore_chan0: ignore the channel 0 of the segmentation (usually for the background)
        """
        super(DiceScore, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.ignore = ignore_chan0

    def forward(self, global_input, global_target, dense_input = False, cutfold = 5):
        """
        Args:
            input: logits, or dense mask otherwise
                    for logits:  with a shape [nz/nb/1, nc, nx, ny]
                    for dense masks: shape [nz/nb/1, 1, nx, ny]
                    cutfold: split the input volume to <cutfold> folds
            target: dense mask instead, always
        """
        assert global_input.dim() == 4
        smooth = 1e-7
        nz = global_input.size(0)
        foldsize = nz // cutfold + 1 #  actual size

        niter = nz // foldsize

        if nz % foldsize == 0:
            pass
        else:
            niter += 1

        assert niter * foldsize >= nz

        global_inter = 0
        global_nz_pred = 0
        global_nz_gth = 0

        # start the loop
        for ii in range( niter ):
            input = global_input[ii * foldsize : (ii + 1) * foldsize, ...].clone()
            target = global_target[ii * foldsize : (ii + 1) * foldsize, ...].clone()

            if dense_input != True:
                # input is logits
                input = F.softmax(input, dim=1).view(-1, self.n_classes) # nxyz, nc
                input = self.one_hot_encoder(torch.argmax(input, 1)) # nxyz, nc
            else:
                input = self.one_hot_encoder( input.view(-1) )


            target = self.one_hot_encoder( target.view(-1)  ) #nxyz, nc

            if self.ignore == True:
                input = input[:,1:, ...]
                target = target[:,1:, ...]

            try:
                inter = torch.sum(input * target, 0) # + smooth # summing over pixel, keep dimension
                nz_pred = torch.sum(input, 0)
                nz_gth = torch.sum(target, 0)

                flat_inter = [] # place holder
            except:
                # magic numbver, probably due to cuda mememory mechanism
                MAGIC_NUMBER = 14000000
                if input.shape[0] < MAGIC_NUMBER:
                    raise ValueError


                flat_inter = input * target
                total_shape = input.shape[0]
                inter = 0
                nz_pred = 0
                nz_gth = 0

                # iterate through it
                for ii in range(total_shape // MAGIC_NUMBER + 1): # python and pytorch allows going over ...
                    inter += torch.sum(flat_inter[MAGIC_NUMBER * ii: MAGIC_NUMBER * (ii+1) ], 0)
                    nz_pred += torch.sum(input[MAGIC_NUMBER * ii: MAGIC_NUMBER * (ii+1) ], 0)
                    nz_gth += torch.sum(target[MAGIC_NUMBER * ii: MAGIC_NUMBER * (ii+1) ], 0)

            del input
            del target
            del flat_inter

            global_inter += inter
            global_nz_pred += nz_pred
            global_nz_gth  += nz_gth

        global_union = global_nz_pred + global_nz_gth + smooth
        score = 2.0 * global_inter / global_union

        return score


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)



class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        """
        input: logits : nb x nc x H x W
        target: dense mask
        """
        smooth = 1e-5
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2)# + smooth # originally there is a smooth on inter which I believe to be a bug!
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score