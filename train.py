import os
import torch
import torch.nn.functional as F
import sys
import time
sys.path.append('models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.model import Network
#from net import DFMNet
from data import get_loader,test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from parsers import opt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# Initialize time tracking variables
total_test_time = 0.0  # <-- New variable
num_epochs_tested = 0  # <-- New variable

model = Network()
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load, map_location=torch.device('cpu')))
    print('load model from ',opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

#set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
#edge_root=opt.edge_root
test_image_root=opt.test_rgb_root
test_gt_root=opt.test_gt_root
#test_edge_root=opt.test_edge_root
save_path=opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CARNet-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))



step=0
writer = SummaryWriter(save_path+'summary')
best_mae=1
best_epoch=0

def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = bce.mean(dim=(2, 3))
    
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    
    return (bce + iou).mean()
'''''
# Variant 1: Weighted BCE only
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()

# Variant 2: Weighted IoU only
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

# Variant 3: Unweighted BCE + IoU
def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = bce.mean(dim=(2, 3))
    
    pred_sig = torch.sigmoid(pred)
    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    
    return (bce + iou).mean()
def structure_loss(pred, mask):
    # Remove weit (boundary-aware weighting)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    
    return (bce + iou).mean()

def edge_loss(pred, gt):
    pred_edge = F.conv2d(pred, torch.ones(1,1,3,3).to(pred.device))
    gt_edge = F.conv2d(gt, torch.ones(1,1,3,3).to(gt.device))
    return F.l1_loss(pred_edge, gt_edge)
'''''
#train function
def train(train_loader, model, optimizer, epoch,save_path):
    #？
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images = images.cuda()
            gts = gts.cuda()

            s1= model(images)
            loss = structure_loss(s1, gts)
            loss.backward()

            clip_gradient(optimizer, opt.clip)

            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s1[0].clone()

                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s1', torch.tensor(res), step,dataformats='HW')

        
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'Net_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
#test function
def test(test_loader,model,epoch,save_path):
    global best_mae, best_epoch, total_test_time, num_epochs_tested, no_improvement_count
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, name,img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res= model(image)
            res = F.interpolate(res, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum+=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
         # Time calculations
        epoch_time = time.time() - start_time  # <-- New
        total_test_time += epoch_time  # <-- New
        num_epochs_tested += 1  # <-- New
        avg_test_time = total_test_time / num_epochs_tested  # <-- New
        mae=mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        writer.add_scalar('Time/Test', epoch_time, global_step=epoch)  # <-- New
        writer.add_scalar('Time/AvgTest', avg_test_time, global_step=epoch)  # <-- New
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = epoch
            no_improvement_count = 0  # Reset counter on first epoch
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model.state_dict(), save_path+'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
                no_improvement_count = 0
            else:
                no_improvement_count += 1    
        logging.info(f'#TEST#: Epoch:{epoch} Time:{epoch_time:.2f}s AvgTime:{avg_test_time:.2f}s MAE:{mae} bestEpoch:{best_epoch} bestMAE:{best_mae}')  # <-- Modified
if __name__ == '__main__':
    print("Start train...")
    no_improvement_count = 0
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
        # Early stopping check
        if no_improvement_count >= 30:
            print(f"No improvement in MAE for 30 consecutive epochs. Early stopping at epoch {epoch}.")
            logging.info(f"Early stopping triggered at epoch {epoch}.")
            break