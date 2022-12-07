import math
import scipy
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from data_RGB import get_training_data, get_validation_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.assn import ASSN
import torch.optim as optim
from model_utils import load_checkpoint,load_optim
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
#import numpy as np
__all__ = [
    'supervised_training_iter',
    'soc_adaptation_iter',
]


# ----------------------------------------------------------------------------------
# Tool Classes/Functions
# ----------------------------------------------------------------------------------

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# ASSN Training Functions
# ----------------------------------------------------------------------------------

blurer = GaussianBlurLayer(1, 3).cuda()


def supervised_training_iter(
    assn, optimizer, image, trimap, gt_matte,
    semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):

    global blurer

    # set the model to train mode and clear the optimizer
    assn.train()
    optimizer.zero_grad()

    # forward the model
    pred_semantic, pred_matte = assn(image, False)

    # calculate the boundary mask from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)

    # calculate the semantic loss
    #print(f'gt_matte{gt_matte.shape}')
    gt_semantic = F.interpolate(gt_matte, scale_factor=1/16, mode='bilinear')
    gt_semantic = blurer(gt_semantic)
    #print(f'pred_semantic{pred_semantic.shape}gt_semantic{gt_semantic.shape}')
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
    semantic_loss = semantic_scale * semantic_loss
    '''
    # calculate the detail loss
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    detail_loss = detail_scale * detail_loss
    '''
    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
        + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss

    #calculate soft-loss


    # calculate the final loss, backward the loss, and update the model
    loss = semantic_loss  + matte_loss
    loss.backward(retain_graph = True)
    optimizer.step()

    # for test
    return semantic_loss, detail_loss, matte_loss

if __name__ == "__main__":
    train_dir = 'E:\\ImageMattingDataset\\SPDResize\\'
    #val_dir = ''
    model_dir = 'E:\\ImageMattingDataset\\checkpoints\\'
    batch_size = 6
    epoches = 800
    #model assn
    assn = ASSN()
    assn.cuda()
    device_ids =  [0]
    if torch.cuda.device_count()>=1:
        print("we use {} for training".format(torch.cuda.device_count()))
    new_lr = 0.0004
    #optimzer
    optimizer = optim.SGD(assn.parameters(),lr = new_lr)

    #加载预训练权重
    if len(device_ids) >=1:

        path_chk_rest = 'E:\\ImageMattingDataset\\checkpoints\\model_epoch_316_loss_20.546175686642528.pth'
        load_checkpoint(assn, path_chk_rest)
        print("this is pre-train model {}".format(path_chk_rest ))
        print("**********************LOADING***********************")
        # start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        load_optim(optimizer, path_chk_rest)

        #assn = nn.DataParallel(assn, device_ids=device_ids)
        #c = torch.load(path_chk_rest)
        #print("cccccccccccccccccccccccccccccccccccc")
        #print(c)
        #assn.load_state_dict(torch.load(path_chk_rest))
        #load_optim(optimizer, path_chk_rest)
        #load_checkpoint(assn, path_chk_rest)
    #Dataloader image
    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size': None})
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True, num_workers=1,
                              drop_last=False, pin_memory=True)

    #val_dataset = get_validation_data(val_dir, {'patch_size': None})
    #val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False,
    #                        pin_memory=True)

    #print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, epoches + 1):
        #每十个epoch lr*0.1
        if epoch!=1 and epoch%10==0:
            new_lr =new_lr*0.1
        #epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        assn.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            #print(
            # zero_grad
            for param in assn.parameters():
                param.grad = None

            gt_matte = data[0].cuda()

            #print("shape {}".format(gt_matte.shape))
            image = data[1].cuda()
            trimap = data[2].cuda()

            # Compute loss at each stage
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(assn, optimizer, image, trimap, gt_matte,
    semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0)
            #semantic_loss, detail_loss, matte_loss * scale
            loss = semantic_loss + detail_loss + matte_loss
            #loss.backward(retain_graph = True)
            #optimizer.step()
            epoch_loss += loss.item()
        print('epoch {} done'.format(epoch))
        torch.save({'epoch': epoch,
                        'state_dict': assn.state_dict(),
                        'optimizer': optimizer.state_dict(),

                        }, os.path.join(model_dir, f"model_epoch_{epoch}_loss_{epoch_loss}.pth"))
def soc_adaptation_iter(
    assn, backup_assn, optimizer, image,
    soc_semantic_scale=100.0, soc_detail_scale=1.0):
    """ Self-Supervised sub-objective consistency (SOC) adaptation iteration of ASSN
    This function fine-tunes ASSN for one iteration in an unlabeled dataset.
    Note that SOC can only fine-tune a converged ASSN, i.e., ASSN that has been
    trained in a labeled dataset.

    Arguments:
        ASSN (torch.nn.Module): instance of ASSN
        backup_ASSN (torch.nn.Module): backup of the trained ASSN
        optimizer (torch.optim.Optimizer): optimizer for self-supervised SOC
        image (torch.autograd.Variable): input RGB image
                                         its pixel values should be normalized
        soc_semantic_scale (float): scale of the SOC semantic loss
                                    NOTE: please adjust according to your dataset
        soc_detail_scale (float): scale of the SOC detail loss
                                  NOTE: please adjust according to your dataset

    Returns:
        soc_semantic_loss (torch.Tensor): loss of the semantic SOC
        soc_detail_loss (torch.Tensor): loss of the detail SOC

    Example:
        import copy
        import torch
        from src.models.ASSN import ASSN
        from src.trainer import soc_adaptation_iter

        bs = 1          # batch size
        lr = 0.00001    # learn rate
        epochs = 10     # total epochs

        assn = torch.nn.DataParallel(ASSN()).cuda()
        assn = LOAD_TRAINED_CKPT()    # NOTE: please finish this function

        optimizer = torch.optim.Adam(assn.parameters(), lr=lr, betas=(0.9, 0.99))
        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            backup_assn = copy.deepcopy(assn)
            for idx, (image) in enumerate(dataloader):
                soc_semantic_loss, soc_detail_loss = \
                    soc_adaptation_iter(assn, backup_assn, optimizer, image)
    """

    global blurer

    # set the backup model to eval mode
    backup_assn.eval()

    # set the main model to train mode and freeze its norm layers
    assn.train()
    assn.module.freeze_norm()

    # clear the optimizer
    optimizer.zero_grad()

    # forward the main model
    pred_semantic, pred_detail, pred_matte = assn(image, False)

    # forward the backup model
    with torch.no_grad():
        _, pred_backup_detail, pred_backup_matte = backup_assn(image, False)

    # calculate the boundary mask from `pred_matte` and `pred_semantic`
    pred_matte_fg = (pred_matte.detach() > 0.1).float()
    pred_semantic_fg = (pred_semantic.detach() > 0.1).float()
    pred_semantic_fg = F.interpolate(pred_semantic_fg, scale_factor=16, mode='bilinear')
    pred_fg = pred_matte_fg * pred_semantic_fg

    n, c, h, w = pred_matte.shape
    np_pred_fg = pred_fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_pred_fg = np_pred_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_pred_fg, size=(side, side))
        eroded = grey_erosion(sample_np_pred_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float().cuda()

    # sub-objectives consistency between `pred_semantic` and `pred_matte`
    # generate pseudo ground truth for `pred_semantic`
    downsampled_pred_matte = blurer(F.interpolate(pred_matte, scale_factor=1/16, mode='bilinear'))
    pseudo_gt_semantic = downsampled_pred_matte.detach()
    pseudo_gt_semantic = pseudo_gt_semantic * (pseudo_gt_semantic > 0.01).float()

    # generate pseudo ground truth for `pred_matte`
    pseudo_gt_matte = pred_semantic.detach()
    pseudo_gt_matte = pseudo_gt_matte * (pseudo_gt_matte > 0.01).float()

    # calculate the SOC semantic loss
    soc_semantic_loss = F.mse_loss(pred_semantic, pseudo_gt_semantic) + F.mse_loss(downsampled_pred_matte, pseudo_gt_matte)
    soc_semantic_loss = soc_semantic_scale * torch.mean(soc_semantic_loss)

    # NOTE: using the formulas in our paper to calculate the following losses has similar results
    # sub-objectives consistency between `pred_detail` and `pred_backup_detail` (on boundaries only)
    backup_detail_loss = boundaries * F.l1_loss(pred_detail, pred_backup_detail, reduction='none')
    backup_detail_loss = torch.sum(backup_detail_loss, dim=(1,2,3)) / torch.sum(boundaries, dim=(1,2,3))
    backup_detail_loss = torch.mean(backup_detail_loss)

    # sub-objectives consistency between pred_matte` and `pred_backup_matte` (on boundaries only)
    backup_matte_loss = boundaries * F.l1_loss(pred_matte, pred_backup_matte, reduction='none')
    backup_matte_loss = torch.sum(backup_matte_loss, dim=(1,2,3)) / torch.sum(boundaries, dim=(1,2,3))
    backup_matte_loss = torch.mean(backup_matte_loss)

    soc_detail_loss = soc_detail_scale * (backup_detail_loss + backup_matte_loss)

    # calculate the final loss, backward the loss, and update the model
    loss = soc_semantic_loss + soc_detail_loss

    loss.backward()
    optimizer.step()

    return soc_semantic_loss, soc_detail_loss

# ----------------------------------------------------------------------------------
