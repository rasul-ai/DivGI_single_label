import sys, json, os, argparse, time
from operator import itemgetter
import shutil
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from models.get_model import get_arch
from utils.get_loaders import get_train_val_cls_loaders, modify_dataset, modify_loader, get_combo_loader

from utils.evaluation import evaluate_multi_cls
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Tuple

# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:

parser.add_argument('--csv_train', type=str, default='data/train_endo_split2.csv', help='path to training data csv')
parser.add_argument('--data_path', type=str, default='data/images/', help='path data')
parser.add_argument('--sampling', type=str, default='instance', help='sampling mode (instance, class, sqrt, prog)')
parser.add_argument('--model_name', type=str, default='bit_resnext50_1', help='architecture')
parser.add_argument('--multi_label_style', type=str, default='hyper_kvasir_a', help='multi label style')
parser.add_argument('--loss_fn', type=str, default='ce', help='loss function (ce)')
parser.add_argument('--do_mixup', type=float, default=0.0, help='mixup coeff (so far only for multi-class)')
parser.add_argument('--do_multigranularities', type=str2bool, nargs='?', const=True, default=False, help='4 training stages')
parser.add_argument('--n_step', type=int, default=4, help='n step')
parser.add_argument('--do_jigsaw', type=str2bool, nargs='?', const=True, default=True, help='generate jigsaw size 8 4 2 1')
parser.add_argument('--do_multilabel', type=str2bool, nargs='?', const=True, default=False, help='4 label levels')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer choice')
parser.add_argument('--patience', type=int, default=2, help='patience before lr reduction')
parser.add_argument('--factor', type=float, default=0.1, help='learning rate decay factor')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--min_lr', type=float, default=-1, help='learning rate (defaults to stopping just after 3rd decay)')
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--n_epochs', type=int, default=7, help='nr epochs') #
parser.add_argument('--metric', type=str, default='mcc', help='which metric to use for monitoring progress (loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')
parser.add_argument('--pretrained_weights', type=str, default=None, help='start from eyepacs-pretrained weights (path to)')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch '
                                                               'for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--n_checkpoints', type=int, default=1, help='nr of best checkpoints to keep (defaults to 3)')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'mcc':
        return operator.gt, 0
    elif metric == 'kappa':
        return operator.gt, 0
    elif metric == 'f1':
        return operator.gt, 0
    elif metric == 'bacc':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)

def cross_entropy_loss(input: torch.Tensor,
                             target: torch.Tensor
                             ) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


def jigsaw_generator_new(images, n):
    if n == 1:
        return images
    else:
        block_size = 512 // n
        permute = torch.randperm(n**2)
        chunks_horizontal = torch.chunk(images, n, 2)
        jigsaws = []
        for chunk in chunks_horizontal:
            jigsaws.extend(torch.chunk(chunk, n, 3))
        jigsaws = torch.stack(jigsaws, 2)
        jigsaws = jigsaws[:, :, permute, :, :]
        jigsaws = jigsaws.reshape(-1, 3, n, n, block_size, block_size)
        jigsaws = jigsaws.permute(0, 1, 2, 4, 3, 5)
        jigsaws = jigsaws.reshape(-1, 3, 512, 512)
        
        return jigsaws

def label2multilabel(labels, class_to_4classes):
    converted_labels = []
    for i in range(4):
        converted_labels.append(labels.new_zeros(labels.shape))
    for batch_idx, label in enumerate(labels):
        multi_label = class_to_4classes[label]
        for i in range(4):
            converted_labels[i][batch_idx] = multi_label[i]
    
    return converted_labels

def run_one_epoch(loader, model, criterion, do_mixup=0., optimizer=None, assess=False):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here
    n_classes = model.n_classes
    number_multi_label = model.number_multi_label if model.do_multilabel else [n_classes]

    if train: model.train()
    else: model.eval()
    if assess:
        probs_all, preds_all, labels_all = [], [], []

    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0

        for i_batch, batch in enumerate(loader):
            if train:
                inputs, labels = batch[0][0], batch[0][1]
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                
                if model.do_multigranularities is True:
                    if model.do_jigsaw is True:
                        n_crops = [8,4,2,1]
                    else:
                        n_crops = [1,1,1,1]
                    n_crops = n_crops[-model.n_step:]
                else:
                    n_crops = [1]
                    
                for stage, n_crop in enumerate(n_crops):
                    # inputs: mixup -> jigsaw
                    lam = 0
                    if do_mixup > 0:
                        lam = np.random.beta(a=do_mixup, b=1)
                        balanced_inputs, balanced_labels = batch[1][0], batch[1][1]
                        balanced_inputs, balanced_labels = balanced_inputs.to(device), balanced_labels.squeeze().to(device)
                        
                        mixed_inputs = (1 - lam) * inputs + lam * balanced_inputs
                        jigsaw_inputs = jigsaw_generator_new(mixed_inputs, n_crop)                    
                    else:
                        jigsaw_inputs = jigsaw_generator_new(inputs, n_crop)
                    
                    # labels: multilabel -> mixup
                    if model.do_multilabel is True:
                        multilabels = label2multilabel(labels, model.class_to_4classes)                        
                        balanced_multilabels = label2multilabel(balanced_labels, model.class_to_4classes) if do_mixup > 0 else None
                    else:
                        multilabels = [labels]
                        balanced_multilabels = [balanced_labels] if do_mixup > 0 else None
                    if do_mixup > 0 and lam > 0:
                        mixed_multilabels = [(1 - lam) * F.one_hot(label, n_class) + lam * F.one_hot(balanced_label, n_class)
                                             for label, balanced_label, n_class in zip(multilabels, balanced_multilabels, number_multi_label)]
                    else:
                        mixed_multilabels = [F.one_hot(label, n_class) for label, n_class in zip(multilabels, number_multi_label)]
                    
                    optimizer.zero_grad()
                    logits = model(jigsaw_inputs, stage=1+stage)
                    loss = 0
                    for pred, label in zip(logits[0], mixed_multilabels):
                        loss += cross_entropy_loss(pred, label)
                    loss.backward()
                    optimizer.step()
                logits = logits[-1][-1].softmax(dim=1)
                
            else:
                inputs, labels = batch[0].to(device), batch[1].squeeze().to(device)
                logits = model(inputs, stage=model.n_step+1)
                if model.do_multigranularities is True:
                    cum = logits[0][-1].softmax(dim=1)
                    for i in range(1, model.n_step):
                        cum += logits[i][-1].softmax(dim=1)
                    logits = cum / model.n_step
                else:
                    logits = logits[0][-1].softmax(dim=1)
                loss = criterion(torch.log(logits), labels)

            if assess:
                probs = logits
                preds = np.argmax(probs.detach().cpu().numpy(), axis=1)
                probs_all.extend(probs.detach().cpu().numpy())
                preds_all.extend(preds)
                labels_all.extend(labels.cpu().numpy())

            # Compute running loss
            running_loss += loss.detach().item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(loss_lr="{:.4f}/{:.8f}".format(float(run_loss), get_lr(optimizer)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    if assess: return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all), run_loss
    return None, None, None, None

def train_model(model, sampling, optimizer, train_criterion, val_criterion, do_mixup, train_loader, val_loader,
                scheduler, metric, n_epochs, exp_path, n_checkpoints):

    best_loss, best_auc, best_bacc, best_k, best_mcc, best_f1, best_epoch, best_models = 10, 0, 0, 0, 0, 0, 0, []
    is_better, best_monitoring_metric = compare_op(metric)
    greater_is_better = best_monitoring_metric == 0
    all_tr_aucs, all_vl_aucs, all_tr_mccs, all_vl_mccs = [], [], [], []
    all_tr_ks, all_vl_ks, all_tr_baccs, all_vl_baccs, all_tr_losses, all_vl_losses = [], [], [], [], [], []

    if model.n_classes == 5: class_names = ['DR0', 'DR1', 'DR2', 'DR3', 'DR4']
    else: class_names = ['C{}'.format(i) for i in range(model.n_classes)]
    print_conf, text_file_train, text_file_val = False, None, None

    for epoch in range(n_epochs):
        print('\nEpoch {:d}/{:d}'.format(epoch+1, n_epochs))
        # Modify sampling
        combo_loader = get_combo_loader(train_loader, base_sampling=sampling)
    
        # train one epoch
        _, _, _, _ = run_one_epoch(combo_loader, model, train_criterion, do_mixup, optimizer, assess=True)

        with torch.no_grad():
            tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch(train_loader, model, val_criterion, assess=True)
            vl_preds, vl_probs, vl_labels, vl_loss = run_one_epoch(val_loader, model, val_criterion, assess=True)


        if exp_path is not None:
            print_conf = True
            text_file_train = osp.join(exp_path,'performance_epoch_{}.txt'.format(str(epoch+1).zfill(2)))
            text_file_val = osp.join(exp_path, 'performance_epoch_{}.txt'.format(str(epoch+1).zfill(2)))

        tr_auc, tr_k, tr_mcc, tr_f1, tr_bacc, tr_auc_all, tr_f1_all = evaluate_multi_cls(tr_labels, tr_preds, tr_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_train, loss=tr_loss)
        vl_auc, vl_k, vl_mcc, vl_f1, vl_bacc, vl_auc_all, vl_f1_all = evaluate_multi_cls(vl_labels, vl_preds, vl_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_val, loss=vl_loss, lr=get_lr(optimizer))

        print('Train||Val Loss: {:.4f}||{:.4f} - K: {:.2f}||{:.2f} - mAUC: {:.2f}||{:.2f} - MCC: {:.2f}||{:.2f} - BACC: {:.2f}||{:.2f}'.format(
            tr_loss, vl_loss, 100 * tr_k, 100 * vl_k, 100 * tr_auc, 100 * vl_auc, 100 * tr_mcc, 100 * vl_mcc, 100 * tr_bacc, 100 * vl_bacc))

        all_tr_aucs.append(tr_auc_all)
        all_vl_aucs.append(vl_auc_all)
        all_tr_mccs.append(tr_mcc)
        all_vl_mccs.append(vl_mcc)
        all_tr_baccs.append(tr_bacc)
        all_vl_baccs.append(vl_bacc)
        all_tr_ks.append(tr_k)
        all_vl_ks.append(vl_k)
        all_tr_losses.append(tr_loss)
        all_vl_losses.append(vl_loss)

        # check if performance was better than anyone before and checkpoint if so
        if metric == 'loss':  tr_monitoring_metric, vl_monitoring_metric  = tr_loss, vl_loss
        elif metric == 'kappa': tr_monitoring_metric, vl_monitoring_metric  =  tr_k, vl_k
        elif metric == 'mcc': tr_monitoring_metric, vl_monitoring_metric  =  tr_mcc, vl_mcc
        elif metric == 'f1':  tr_monitoring_metric, vl_monitoring_metric  =  tr_f1, vl_f1
        elif metric == 'auc': tr_monitoring_metric, vl_monitoring_metric  =  tr_auc, vl_auc
        elif metric == 'bacc': tr_monitoring_metric, vl_monitoring_metric = tr_bacc, vl_bacc

        # if tr_monitoring_metric > vl_monitoring_metric:  # only if we do not underfit
        #     scheduler.step(vl_monitoring_metric)
        scheduler.step()

        if is_better(vl_monitoring_metric, best_monitoring_metric):
            print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, 100*best_monitoring_metric, 100*vl_monitoring_metric))
            best_loss, best_k, best_mcc, best_f1, best_auc, best_bacc, best_epoch = vl_loss, vl_k, vl_mcc, vl_f1, vl_auc, vl_bacc, epoch+1
            best_monitoring_metric = vl_monitoring_metric
        else:
            print('-------- Best {} so far {:.2f} at epoch {:d} --------'.format(metric, 100 * best_monitoring_metric,
                                                                                 best_epoch))

        # SAVE n best - keep deleting worse ones
        if exp_path is not None:
            s_name = 'epoch_{}_K_{:.2f}_mAUC_{:.2f}_MCC_{:.2f}'.format(str(epoch + 1).zfill(2), 100 * vl_k,
                                                                       100 * vl_auc, 100 * vl_mcc)
            best_models.append([osp.join(exp_path, s_name), vl_monitoring_metric])

            if epoch < n_checkpoints:  # first n_checkpoints epochs save always
                print('-------- Checkpointing to {}/ --------'.format(s_name))
                save_model(osp.join(exp_path, s_name), model, optimizer, weights=True)
            else:
                worst_model = sorted(best_models, key=itemgetter(1), reverse=greater_is_better)[-1][
                    0]  # False for Loss, True for K
                if s_name != worst_model:  # this model was better than one of the best n_checkpoints models, remove that one
                    print('-------- Checkpointing to {}/ --------'.format(s_name))
                    save_model(osp.join(exp_path, s_name), model, optimizer, weights=True)
                    # print('before deleting', os.listdir(osp.join(exp_path, s_name)))
                    print('----------- Deleting {}/ -----------'.format(worst_model.split('/')[-1]))
                    shutil.rmtree(worst_model)
                    best_models = sorted(best_models, key=itemgetter(1), reverse=greater_is_better)[:n_checkpoints]

        # if get_lr(optimizer) <= scheduler.min_lrs[0]:
        #     print('Early stopping')
        #     del model
        #     torch.cuda.empty_cache()
        #     return best_auc, best_bacc, best_mcc, best_k, all_tr_aucs, all_vl_aucs, all_tr_mccs, all_vl_mccs, \
        #            all_tr_ks, all_vl_ks, all_tr_losses, all_vl_losses, best_epoch

    del model
    torch.cuda.empty_cache()
    return best_auc, best_bacc, best_mcc, best_k, all_tr_aucs, all_vl_aucs, all_tr_mccs, all_vl_mccs, \
           all_tr_ks, all_vl_ks, all_tr_losses, all_vl_losses, best_epoch


if __name__ == '__main__':

    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # reproducibility
    seed_value = 1243
    set_seeds(seed_value, use_cuda)

    data_path = args.data_path

    # gather parser parameters
    sampling = args.sampling
    model_name = args.model_name
    multi_label_style = args.multi_label_style
    optimizer_choice = args.optimizer
    lr, min_lr, factor, patience, bs = args.lr, args.min_lr, args.factor, args.patience, args.batch_size
    if min_lr == -1: min_lr = lr * 1e-3

    n_epochs, metric = args.n_epochs, args.metric

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path=osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)
        n_checkpoints = args.n_checkpoints
        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path, n_checkpoints=None, 0

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    print('* Instantiating a {} model.'.format(model_name))
    do_multigranularities = str2bool(args.do_multigranularities)
    n_step = args.n_step
    do_jigsaw = str2bool(args.do_jigsaw)
    do_multilabel = str2bool(args.do_multilabel)
    if do_multigranularities is True:
        print("* Add multi-granularities with {} steps.".format(n_step))
    else:
        n_step = 1
        
    if do_multilabel is True:
        print("* Add multi-labels.")
    number_multi_label = {'hyper_kvasir_a': [2, 7, 13, 23], 'hyper_kvasir_tree': [2, 6, 15, 23], 'upper_gi_a': [2, 3, 9, 12], 'gastrovision_a': [3, 11, 15, 22], 'gastrovision_tree': [3, 9, 22, 22]}
    model, mean, std = get_arch(model_name, n_classes=number_multi_label[multi_label_style][-1], do_multigranularities=do_multigranularities, n_step=n_step,
                                do_multilabel=do_multilabel, number_multi_label=number_multi_label[multi_label_style])
    model.do_jigsaw = do_jigsaw
    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_cls_loaders(csv_path_train=csv_train, csv_path_val=csv_val,
                                                         data_path=data_path, batch_size=bs,
                                                         tg_size=tg_size, mean=mean, std=std,
                                                         num_workers=args.num_workers)
    class_to_4classes = {'hyper_kvasir_a': [[0, 0, 0, 0],
                                        [0, 0, 0, 1],
                                        [0, 0, 1, 2],
                                        [0, 0, 1, 3],
                                        [0, 0, 2, 4],
                                        [0, 1, 3, 5],
                                        [0, 2, 4, 6],
                                        [1, 3, 5, 7],
                                        [1, 3, 5, 8],
                                        [1, 3, 5, 9],
                                        [1, 3, 6, 10],
                                        [1, 3, 6, 11],
                                        [1, 3, 7, 12],
                                        [1, 3, 8, 13],
                                        [1, 3, 8, 14],
                                        [1, 3, 8, 15],
                                        [1, 3, 8, 16],
                                        [1, 3, 8, 17],
                                        [1, 3, 8, 18],
                                        [1, 4, 9, 19],
                                        [1, 4, 10, 20],
                                        [1, 5, 11, 21],
                                        [1, 6, 12, 22],],
                         'hyper_kvasir_tree': [[0, 0, 0, 0],
                                        [0, 0, 0, 1],
                                        [0, 0, 1, 2],
                                        [0, 0, 1, 3],
                                        [0, 1, 2, 4],
                                        [0, 1, 3, 5],
                                        [0, 1, 4, 6],
                                        [1, 2, 5, 7],
                                        [1, 2, 5, 8],
                                        [1, 2, 6, 9],
                                        [1, 3, 7, 10],
                                        [1, 3, 8, 11],
                                        [1, 4, 9, 12],
                                        [1, 4, 10, 13],
                                        [1, 4, 10, 14],
                                        [1, 4, 10, 15],
                                        [1, 4, 10, 16],
                                        [1, 4, 10, 17],
                                        [1, 4, 10, 18],
                                        [1, 4, 11, 19],
                                        [1, 5, 12, 20],
                                        [1, 5, 13, 21],
                                        [1, 5, 14, 22],],
                            'upper_gi_a': [[0, 0, 0, 0],
                                        [1, 1, 1, 1],
                                        [1, 1, 2, 2],
                                        [1, 1, 2, 3],
                                        [1, 2, 3, 4],#'4-fundus'
                                        [1, 1, 4, 5],#'5-middle-upper body antegrade view'
                                        [1, 2, 5, 6],#'6-middle-upper body retroflex view'
                                        [1, 2, 6, 7],#'7-angulus'
                                        [1, 1, 4, 8],#'8-lower body'
                                        [1, 1, 7, 9],
                                        [1, 1, 8, 10],
                                        [1, 1, 8, 11]],
                            'gastrovision_a': [
                                        [0, 0, 0, 0],
                                        [0, 1, 1, 1],
                                        [1, 2, 2, 2],
                                        [1, 2, 2, 3],
                                        [1, 2, 3, 4],
                                        [1, 2, 3, 5],
                                        [1, 3, 4, 6],
                                        [1, 4, 5, 7],
                                        [1, 5, 6, 8],
                                        [1, 6, 7, 9],
                                        [2, 7, 8, 10],
                                        [2, 7, 9, 11],
                                        [2, 8, 10, 12],
                                        [2, 8, 10, 13],
                                        [2, 8, 10, 14],
                                        [2, 8, 10, 15],
                                        [2, 8, 11, 16],
                                        [2, 8, 12, 17],
                                        [2, 8, 12, 18],
                                        [2, 8, 12, 19],
                                        [2, 9, 13, 20],
                                        [2, 10, 14, 21]],
                            'gastrovision_tree': [
                                        [0, 0, 0, 0],
                                        [0, 1, 1, 1],
                                        [1, 2, 2, 2],
                                        [1, 2, 2, 3],
                                        [1, 3, 4, 4],
                                        [1, 4, 5, 5],
                                        [1, 3, 6, 6],
                                        [1, 2, 7, 7],
                                        [1, 4, 8, 8],
                                        [1, 3, 9, 9],
                                        [2, 6, 10, 10],
                                        [2, 6, 11, 11],
                                        [2, 6, 12, 12],
                                        [2, 7, 13, 13],
                                        [2, 7, 14, 14],
                                        [2, 7, 15, 15],
                                        [2, 5, 16, 16],
                                        [2, 8, 17, 17],
                                        [2, 8, 18, 18],
                                        [2, 8, 19, 19],
                                        [2, 6, 20, 20],
                                        [2, 6, 21, 21]],} 
    model.class_to_4classes = class_to_4classes[multi_label_style]
    
    model = model.to(device)

    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        to_optimise = [{'params': model.model.parameters(), 'lr': lr * 0.1}] + [{'params': getattr(model, param_name).parameters(), 'lr': lr} for param_name in model.params_to_learn]
        optimizer = torch.optim.SGD(to_optimise, momentum=0.9, weight_decay=5e-4)
    else:
        sys.exit('please choose a valid optimizer')

    if args.pretrained_weights is True:
        weights_path = osp.join('data/pretrained_weights/', model_name)
        try:
            model, stats, optimizer_state_dict = load_model(model, args.resume_path, device=device, with_opt=True)
            optimizer.load_state_dict(optimizer_state_dict)
        except:
            sys.exit('Pretrained weights not compatible for this model')
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr

    scheduler = CosineAnnealingLR(optimizer, n_epochs)


    train_criterion, val_criterion = torch.nn.NLLLoss(), torch.nn.NLLLoss()
    do_mixup = args.do_mixup

    print('* Instantiating loss function', str(train_criterion))
    print('* Starting to train\n','-' * 10)
    start = time.time()
    b_mauc, b_bacc, b_mcc, b_k, tr_aucs, vl_aucs, \
    tr_mccs, vl_mccs, tr_ks, vl_ks, tr_ls, vl_ls, b_epoch = train_model(model, sampling, optimizer, train_criterion, val_criterion,
                                                                        do_mixup, train_loader, val_loader, scheduler,
                                                                        metric, n_epochs, experiment_path, n_checkpoints)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("b_mauc: %f" % b_mauc)
    print("b_mcc: %f" % b_mcc)
    print("b_k: %f" % b_k)
    print("b_epoch: %d" % b_epoch)

    if do_not_save is False:
        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print(
                'Best K = {:.2f}\nBest mAUC = {:.2f}\nBest MCC = {:.2f}\nBest BACC = {:.2f}\nBest epoch = {}\n'.format(
                    100 * b_k, 100 * b_mauc, 100 * b_mcc, 100 * b_bacc, b_epoch), file=f)
            for j in range(len(vl_aucs)):
                print(
                    'Epoch = {} -> K={:.2f}/{:.2f}, mAUC={:.2f}/{:.2f}, MCC={:.2f}/{:.2f}, Loss={:.4f}/{:.4f},'.format(
                        j + 1, 100 * tr_ks[j], 100 * vl_ks[j],
                        100 * np.mean(tr_aucs[j]), 100 * np.mean(vl_aucs[j]),
                        100 * tr_mccs[j], 100 * vl_mccs[j], tr_ls[j], vl_ls[j]), file=f)

            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
