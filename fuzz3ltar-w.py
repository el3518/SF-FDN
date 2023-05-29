import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import networkl as network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.utils.data as data_utils
import torch.nn.functional as F

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load_tar(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_tst(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i       
        
        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()
    
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def srcnet_output(inputs, netF, netB, netC, cen, args):
    
    feas = netB(netF(inputs))
    mem_ship = clu_mem(feas.detach().cpu(), cen, args)
    tar_rule = np.argsort(-mem_ship, axis=1)
    mem_ship = torch.from_numpy(mem_ship).float().cuda()            
    outputs = netC(feas) 
        
    return mem_ship, tar_rule, outputs

def cal_weight(inputs, outputs, netF, netB, netG, init_ent, weights_all):
    
    features = netB(netF(inputs))
    softmax_ = nn.Softmax(dim=1)(outputs)
    #ent_loss = torch.mean(loss.Entropy(softmax_ ))
    #init_en = ent_loss
    weights_test = netG_list(features)
    weights_all = weights_test.squeeze() 
        
    return weights_all#init_ent, 

  
def cal_globe(loader, netF, netB1, netC1, netB2, netC2, netB3, netC3, netG_list, cen1, cen2, cen3, args, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            mem_ship1, tar_rule1, outputs1 = srcnet_output(inputs, netF, netB1, netC1, cen1, args)           
            mem_ship2, tar_rule2, outputs2 = srcnet_output(inputs, netF, netB2, netC2, cen2, args)           
            mem_ship3, tar_rule3, outputs3 = srcnet_output(inputs, netF, netB3, netC3, cen3, args)           
                        
            outputs_all = torch.zeros(args.sk, inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], args.sk)
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)
            
            outputs_all[0] = cal_output_sel(outputs1, mem_ship1, tar_rule1, args.rule_sel1, args)
            outputs_all[1] = cal_output_sel(outputs2, mem_ship2, tar_rule2, args.rule_sel2, args)
            outputs_all[2] = cal_output_sel(outputs3, mem_ship3, tar_rule3, args.rule_sel3, args)
            
            #init_ent[:,0], weights_all[:, 0] = cal_weight(inputs, outputs, netF, netB, netG, init_ent, weights_all)
            
            features = netB1(netF(inputs))
            weights_test = netG_list[0](features)
            weights_all[:, 0] = weights_test.squeeze()
            
            features = netB2(netF(inputs))
            weights_test = netG_list[1](features)
            weights_all[:, 1] = weights_test.squeeze()
            
            features = netB3(netF(inputs))
            weights_test = netG_list[2](features)
            weights_all[:, 2] = weights_test.squeeze()
            
            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
            print(weights_all.mean(dim=0))
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])

            outputs = outputs_all_w
            #outputs = (outputs1+outputs2+outputs3)/3
            #############################################################
            '''
            weights = nn.Softmax(dim=1)(netW(outputs))
            w1 = weights[:, 0].reshape(outputs.shape[0],1)
            w2 = weights[:, 1].reshape(outputs.shape[0],1)
            w3 = weights[:, 2].reshape(outputs.shape[0],1)
            outputs = w1*outputs1 + w2*outputs2 + w3*outputs3
            '''
            
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:              
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
      
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc(loader, netF, netB, netC, cen, args, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            mem_ship, tar_rule, outputs = srcnet_output(inputs, netF, netB, netC, cen, args)           
            
            outputs = cal_output(outputs, mem_ship, args) #predicted            
                     
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def clu_mem(fea, cen, args):
    
    if args.distance == 'cosine':
        fea = norm_fea(fea).numpy()
    dist_c = cdist(fea, cen, args.distance)
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(cen.shape[0], axis=1)
    #mem_ship = nn.Softmax(dim=1)(torch.from_numpy(1/(1e-8 + (dist_c*dda)))).numpy() 
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy()     
        
    return mem_ship


def clu_cen_lab_(loader, netF, netB, netC, cen, src_rule_num, args):#class center, label
    start_test = True #loader = dset_loaders["test"]
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            
            feas = netB(netF(inputs))
            
            mem_ship, tar_rule, outputs = srcnet_output(inputs, netF, netB, netC, cen, args)           
            
            outputs = cal_output_sel(outputs, mem_ship, tar_rule, src_rule_num, args)
                               
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
  
    all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    probs, predict = torch.max(all_output, 1)
    
    #idex = sel_sam(probs.float().numpy(), predict.float().numpy().astype('int'), args)
                
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    #mem_ship = clu_mem(all_fea, cen, args)
    
    #if args.distance == 'cosine':
    #    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    #    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    
    
    if args.da == 'oda':
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
        ent = ent.float().cpu()

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))
        
        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        known_idx = np.where(kmeans.labels_ != iidx)[0]

        all_fea = all_fea[known_idx,:]
        all_output = all_output[known_idx,:]
        predict = predict[known_idx]
        all_label_idx = all_label[known_idx]
        ENT_THRESHOLD = (kmeans.cluster_centers_).mean()               
        
        idex = sel_sam(probs[known_idx].float().numpy(), predict.float().numpy().astype('int'), args)    
        dist_c = cdist(all_fea, cen, args.distance)
        dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
        dist_a = np.expand_dims(dist_a, axis=1)
        dda=dist_a.repeat(cen.shape[0], axis=1)
        mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy() 
        all_fea = all_fea.float().cpu().numpy()
    else:
        idex = sel_sam(probs.float().numpy(), predict.float().numpy().astype('int'), args)            
        mem_ship = clu_mem(all_fea, cen, args)
        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()
        
    #all_fea = all_fea.float().cpu().numpy()
    ##rule cen #############################################
    #nn.Softmax(dim=1)(torch.from_numpy(np.power(mem_ship,2))).numpy()
    aff = np.power(mem_ship,2)#mem_ship #######################
    initc_rule = aff.transpose().dot(all_fea)
    initc_rule = initc_rule / (1e-8 + aff.sum(axis=0)[:,None]) 
        
    ## anchor #############################################
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy() 
    ###################################    
    
    initc_all = aff.transpose().dot(all_fea)
    initc_all = initc_all / (1e-8 + aff.sum(axis=0)[:,None])
    
    cls_count = np.eye(K)[predict].sum(axis=0) #class number 
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    ######################################
    '''
    dist_c = cdist(all_fea, initc_all[labelset], args.distance)#
    dist_a = cdist(all_fea, initc_all[labelset], args.distance).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(len(labelset), axis=1)
    mem_ship = nn.Softmax(dim=1)(torch.from_numpy(1/(1e-8 + (dist_c/dda)))).numpy()    
    pred_label = mem_ship.argmax(axis=1)
    pred_label = labelset[pred_label]
    '''
    #########################################

    initc, pred_label_cof = update_cen_lab(all_fea, initc_all, predict, labelset, K, idex, args)     

    idex_same = np.where(pred_label_cof == predict.numpy().astype('int'))[0]
    idex_u = list(set(idex) & set(idex_same))
    
    initc, pred_label = update_cen_lab(all_fea, initc, predict, labelset, K, idex_u, args) 
    
    pred_label[idex_u] = pred_label_cof[idex_u]
    
    if args.da == 'oda':
        guess_label = args.class_num * np.ones(len(all_label), )
        guess_label[known_idx] = pred_label

        acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label)
        log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)
        
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str +'\n')

        return initc_rule, guess_label.astype('int'), ENT_THRESHOLD
    else:        
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str +'\n')

        return initc_rule, pred_label, labelset
################################################
def clu_cen_lab(loader, netF, netB, netC, cen, src_rule_num, args):#class center, label
    start_test = True #loader = dset_loaders["test"]
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            
            feas = netB(netF(inputs))
            
            mem_ship, tar_rule, outputs = srcnet_output(inputs, netF, netB, netC, cen, args)           
            
            outputs = cal_output_sel(outputs, mem_ship, tar_rule, src_rule_num, args)
                               
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
  
    all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    probs, predict = torch.max(all_output, 1)
    
    #idex = sel_sam(probs.float().numpy(), predict.float().numpy().astype('int'), args)
                
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    #mem_ship = clu_mem(all_fea, cen, args)
    
    #if args.distance == 'cosine':
    #    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    #    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    
    
    if args.da == 'oda':
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
        ent = ent.float().cpu()

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))
        
        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        known_idx = np.where(kmeans.labels_ != iidx)[0]

        all_fea = all_fea[known_idx,:]
        all_output = all_output[known_idx,:]
        predict = predict[known_idx]
        all_label_idx = all_label[known_idx]
        ENT_THRESHOLD = (kmeans.cluster_centers_).mean()               
        
        idex = sel_sam(probs[known_idx].float().numpy(), predict.float().numpy().astype('int'), args)    
        dist_c = cdist(all_fea, cen, args.distance)
        dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
        dist_a = np.expand_dims(dist_a, axis=1)
        dda=dist_a.repeat(cen.shape[0], axis=1)
        mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy() 
        all_fea = all_fea.float().cpu().numpy()
    else:
        idex = sel_sam(probs.float().numpy(), predict.float().numpy().astype('int'), args)            
        mem_ship = clu_mem(all_fea, cen, args)
        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()
        
    ## anchor #############################################
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy() 
    ###################################    
    
    initc_all = aff.transpose().dot(all_fea)
    initc_all = initc_all / (1e-8 + aff.sum(axis=0)[:,None])
    
    cls_count = np.eye(K)[predict].sum(axis=0) #class number 
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    ######################################
    '''
    dist_c = cdist(all_fea, initc_all[labelset], args.distance)#
    dist_a = cdist(all_fea, initc_all[labelset], args.distance).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(len(labelset), axis=1)
    mem_ship = nn.Softmax(dim=1)(torch.from_numpy(1/(1e-8 + (dist_c/dda)))).numpy()    
    pred_label = mem_ship.argmax(axis=1)
    pred_label = labelset[pred_label]
    '''
    #########################################

    initc, pred_label_cof = update_cen_lab(all_fea, initc_all, predict, labelset, K, idex, args)     

    idex_same = np.where(pred_label_cof == predict.numpy().astype('int'))[0]
    idex_u = list(set(idex) & set(idex_same))
    
    initc, pred_label = update_cen_lab(all_fea, initc, predict, labelset, K, idex_u, args) 
    
    pred_label[idex_u] = pred_label_cof[idex_u]
    
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        
    '''
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    '''
    print(log_str +'\n')
    #return initc_rule, pred_label, labelset
    return initc, all_fea
###############################################
def sel_sam(prob, lab, args):
    
    idex = []
    for cls in range(args.class_num):
        idx = [idx for idx, lab in enumerate(lab) if lab == cls]           
        if len(idx) > 0:
            idxs = [idxs for idxs in idx if prob[idxs] >= np.median(prob[idx])]#-(np.median(prob[idx])-np.min(prob[idx]))/(args.sk+1)]
            idex.extend(idxs)    
    return idex

def update_cen_lab(all_fea, initc, predict, labelset, K, idex, args):
            
    confi_pre = predict[idex].float().cpu().int().numpy()
    class_tup = []
    for i in confi_pre:
        if i not in class_tup:
            class_tup.append(i)
    
    aff_confi = np.eye(K)[confi_pre]    
    initc0 = aff_confi.transpose().dot(all_fea[idex])                
    initc0 = initc0 / (1e-8 + aff_confi.sum(axis=0)[:,None])
    for i in range(args.class_num):
        if i not in class_tup:
            initc0[i] = initc[i]
        
    dist_c = cdist(all_fea[idex], initc0[labelset], args.distance)#
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(len(labelset), axis=1)
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy()
    pred_label = mem_ship.argmax(axis=1)
    pred_label = labelset[pred_label]
    
    for round in range(1):
        aff = np.eye(K)[pred_label]#mem_ship#
        initc1 = aff.transpose().dot(all_fea[idex])
        initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])
        for i in range(args.class_num):
            if i not in class_tup:
                initc1[i] = initc0[i]
        dist_c = cdist(all_fea, initc1[labelset], args.distance)
        dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
        dist_a = np.expand_dims(dist_a, axis=1)
        dda=dist_a.repeat(len(labelset), axis=1)
        #mem_ship = nn.Softmax(dim=1)(torch.from_numpy(1/(1e-8 + (dist_c*dda)))).numpy() 
        mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy()     
        pred_label = mem_ship.argmax(axis=1)
        pred_label = labelset[pred_label]    
    
    return initc1, pred_label.astype('int')

def clu_cen(loader, netF, netB, netC, cen, src_rule_num, args):#fuzzy cluster
    start_test = True #loader = dset_loaders["test"]
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            
            feas = netB(netF(inputs))
            
            mem_ship, tar_rule, outputs = srcnet_output(inputs, netF, netB, netC, cen, args)           
            
            outputs = cal_output_sel(outputs, mem_ship, tar_rule, src_rule_num, args)
                               
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
  
    all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    probs, predict = torch.max(all_output, 1)
    
    #idex = sel_sam(probs.float().numpy(), predict.float().numpy().astype('int'), args)
                
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    #mem_ship = clu_mem(all_fea, cen, args)
    
    #if args.distance == 'cosine':
    #    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    #    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    
    if args.da == 'oda':
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
        ent = ent.float().cpu()

        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        known_idx = np.where(kmeans.labels_ != iidx)[0]

        all_fea = all_fea[known_idx,:]
        all_output = all_output[known_idx,:]
        predict = predict[known_idx]
        all_label_idx = all_label[known_idx]
        ENT_THRESHOLD = (kmeans.cluster_centers_).mean()
        
        idex = sel_sam(probs[known_idx].float().numpy(), predict.float().numpy().astype('int'), args)    
        dist_c = cdist(all_fea, cen, args.distance)
        dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
        dist_a = np.expand_dims(dist_a, axis=1)
        dda=dist_a.repeat(cen.shape[0], axis=1)
        mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy() 
        all_fea = all_fea.float().cpu().numpy()
    else:
        idex = sel_sam(probs.float().numpy(), predict.float().numpy().astype('int'), args)            
        mem_ship = clu_mem(all_fea, cen, args)
        if args.distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()
        
    #all_fea = all_fea.float().cpu().numpy()
    ##rule cen #############################################
    #nn.Softmax(dim=1)(torch.from_numpy(np.power(mem_ship,2))).numpy()
    aff = np.power(mem_ship,2)#mem_ship ##########################
    initc_rule = aff.transpose().dot(all_fea)
    initc_rule = initc_rule / (1e-8 + aff.sum(axis=0)[:,None]) 

    return initc_rule

def cal_lab_tra(inputs, netF, netB, netC, cen, args):
    
    mem_ship, _, outputs = srcnet_output(inputs, netF, netB, netC, cen, args)                       
    outputs = cal_output(outputs, mem_ship, args)
                   			
    return outputs

def cal_output(output, member, args):    
    outputs = torch.zeros([output[0].shape[0], args.class_num]).cuda()
    for i in range(len(output)):                       
        outputs += member[:,i].reshape(output[0].shape[0],1)*output[i]               			
    return outputs
    
def cal_output_sel(output, member, tar_rule, src_rule_num, args):    
    outputs = torch.zeros([output[0].shape[0], args.class_num]).cuda()
    for i in range(src_rule_num): #rule_num
        for j in range(output[0].shape[0]):
            outputs[j,:] = outputs[j,:] + member[j,tar_rule[j,i]]*output[tar_rule[j,i]][j,:]              			
    return outputs

def norm_fea(fea):    
    fea = torch.cat((fea, torch.ones(fea.size(0), 1)), 1)
    fea = (fea.t() / torch.norm(fea, p=2, dim=1)).t()   
    return fea


def cal_anc_tra(inputs, netF, netB, output, args):
    
    #cen = np.zeros([])
    fea = netB(netF(inputs)).detach().cpu()
    if args.distance == 'cosine':
        fea = norm_fea(fea).float().numpy()
    output = nn.Softmax(dim=1)(output)
    aff = output.detach().float().cpu().numpy()
    initc = aff.transpose().dot(fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            	   
    return initc

def train_target(args):

    dset_loaders = data_load_tar(args)
       
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    w = 2*torch.rand((args.sk,))-1
    print(w)
    
    netB1 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num = args.rule_num1).cuda()
    netB2 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num = args.rule_num2).cuda()       
    netB3 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC3 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num = args.rule_num3).cuda()
      
    ##################################################################
    #netW = nn.Linear(args.class_num, args.sk).cuda()  
    netG_list = [network.scalar(w[i]).cuda() for i in range(args.sk)]     
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/' + args.name_src1+'_B.pt'   
    netB1.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src1+'_C.pt'   
    netC1.load_state_dict(torch.load(args.modelpath))
    
    args.modelpath = args.output_dir_src + '/' + args.name_src2+'_B.pt'   
    netB2.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src2+'_C.pt'   
    netC2.load_state_dict(torch.load(args.modelpath))  
    
    args.modelpath = args.output_dir_src + '/' + args.name_src3+'_B.pt'   
    netB3.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/' + args.name_src3+'_C.pt'   
    netC3.load_state_dict(torch.load(args.modelpath))  
    
    netC1.eval()
    netC2.eval()
    netC3.eval()
    
    for k, v in netC1.named_parameters():
        v.requires_grad = False
    for k, v in netC2.named_parameters():
        v.requires_grad = False
    for k, v in netC3.named_parameters():
        v.requires_grad = False
    ########
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    
    for k, v in netB1.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    
    for k, v in netB2.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    
    for k, v in netB3.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    
    ###############################
    '''
    for k, v in netW.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr}]
        else:
            v.requires_grad = False
    '''
    ###############################
    for i in range(args.sk):
        for k, v in netG_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr}]
    ############################

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    src_rule_num1 = args.rule_sel1
    src_rule_num2 = args.rule_sel2
    src_rule_num3 = args.rule_sel3
    
    max_epo = len(dset_loaders["target"])    
    max_iter = args.max_epoch * max_epo
    
    interval_iter = max_iter // args.interval
    iter_num = 0
    
    ini_cen1 = np.load(args.output_dir_src+"/"+args.name_src1+"_cen.npy")
    ini_cen2 = np.load(args.output_dir_src+"/"+args.name_src2+"_cen.npy")
    ini_cen3 = np.load(args.output_dir_src+"/"+args.name_src3+"_cen.npy")
    
    src_anc1 = np.load(args.output_dir_src+"/"+args.name_src1+"_anc_para.npy")#_para
    src_anc2 = np.load(args.output_dir_src+"/"+args.name_src2+"_anc_para.npy")
    src_anc3 = np.load(args.output_dir_src+"/"+args.name_src3+"_anc_para.npy")
    
        
    while iter_num < max_iter:
        
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()
            
        if inputs_test.size(0) == 1:# or inputs_target.size(0) == 1:
            continue
       
        if iter_num % interval_iter == 0 and args.cls_par > 0:#max_iter/2:
            netF.eval()
            netB1.eval()
            netC1.eval()
            netB2.eval()
            netC2.eval()  
            netB3.eval()
            netC3.eval()   #ENT_THRESHOLD 
            for i in range(args.sk):
                netG_list[i].eval() 
            
            initc = []
            all_feas = []                          
            
            temp1, temp2 = clu_cen_lab(dset_loaders['test'], netF, netB1, netC1, ini_cen1, src_rule_num1, args) ##pseudo            
            temp1 = torch.from_numpy(temp1).cuda()
            temp2 = torch.from_numpy(temp2).cuda()
            initc.append(temp1)
            all_feas.append(temp2)          

            temp1, temp2 = clu_cen_lab(dset_loaders['test'], netF, netB2, netC2, ini_cen2, src_rule_num2, args) ##pseudo            
            temp1 = torch.from_numpy(temp1).cuda()
            temp2 = torch.from_numpy(temp2).cuda()
            initc.append(temp1)
            all_feas.append(temp2) 

            temp1, temp2 = clu_cen_lab(dset_loaders['test'], netF, netB3, netC3, ini_cen3, src_rule_num3, args) ##pseudo            
            temp1 = torch.from_numpy(temp1).cuda()
            temp2 = torch.from_numpy(temp2).cuda()
            initc.append(temp1)
            all_feas.append(temp2)
                        
            netF.train()
            netB1.train()
            netB2.train()
            netB3.train()
            for i in range(args.sk):
                netG_list[i].train()
                        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_test = inputs_test.cuda()       
                                                         
        '''
        outputs1 = cal_lab_tra(inputs_test, netF, netB1, netC1, ini_cen1, args)
        outputs2 = cal_lab_tra(inputs_test, netF, netB2, netC2, ini_cen2, args)
        outputs3 = cal_lab_tra(inputs_test, netF, netB3, netC3, ini_cen3, args)
        '''       
        outputs_all = torch.zeros(args.sk, inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], args.sk)
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)
        init_ent = torch.zeros(1,args.sk)
        
        #idx = 0
        outputs_all[0] = cal_lab_tra(inputs_test, netF, netB1, netC1, ini_cen1, args)
        features_test = netB1(netF(inputs_test))
        softmax_ = nn.Softmax(dim=1)(outputs_all[0])
        ent_loss = torch.mean(loss.Entropy(softmax_ ))
        init_ent[:,0] = ent_loss
        weights_test = netG_list[0](features_test)
        weights_all[:, 0] = weights_test.squeeze()
        
        outputs_all[1] = cal_lab_tra(inputs_test, netF, netB2, netC2, ini_cen2, args)
        features_test = netB2(netF(inputs_test))
        softmax_ = nn.Softmax(dim=1)(outputs_all[1])
        ent_loss = torch.mean(loss.Entropy(softmax_ ))
        init_ent[:,1] = ent_loss
        weights_test = netG_list[1](features_test)
        weights_all[:, 1] = weights_test.squeeze()
        
        outputs_all[2] = cal_lab_tra(inputs_test, netF, netB3, netC3, ini_cen3, args)
        features_test = netB3(netF(inputs_test))
        softmax_ = nn.Softmax(dim=1)(outputs_all[2])
        ent_loss = torch.mean(loss.Entropy(softmax_ ))
        init_ent[:,2] = ent_loss
        weights_test = netG_list[2](features_test)
        weights_all[:, 2] = weights_test.squeeze()
        

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)
        
        z_2 = torch.sum(weights_all)
        z_ = z_/z_2
    
        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])
        
        
        outputs = outputs_all_w
        #outputs = (outputs1+outputs2+outputs3)/3
        '''
        #################################################
        #outputs = torch.stack([outputs1, outputs2, outputs3], 2)
        weights = nn.Softmax(dim=1)(netW(outputs))
        w1 = weights[:, 0].reshape(outputs.shape[0],1)
        w2 = weights[:, 1].reshape(outputs.shape[0],1)
        w3 = weights[:, 2].reshape(outputs.shape[0],1)
        outputs = w1*outputs1 + w2*outputs2 + w3*outputs3
        #outputs = torch.squeeze(outputs,2)
        #################################################
        '''
        
        tar_anc1 = cal_anc_tra(inputs_test, netF, netB1, outputs, args)#(outputs1+outputs2)/2
        tar_anc2 = cal_anc_tra(inputs_test, netF, netB2, outputs, args)
        tar_anc3 = cal_anc_tra(inputs_test, netF, netB3, outputs, args)
                
        cen_loss1 = np.linalg.norm(src_anc1-tar_anc1, ord=2, keepdims=True) 
        cen_loss2 = np.linalg.norm(src_anc2-tar_anc2, ord=2, keepdims=True) 
        cen_loss3 = np.linalg.norm(src_anc3-tar_anc3, ord=2, keepdims=True) 
                                                   
        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(args.sk):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs, pred.cpu())
            #############################################
            '''
            pred1 = clu_label1[tar_idx]
            classifier_losst1 = nn.CrossEntropyLoss()(outputs, pred1) #
            classifier_losst1 *= args.cls_par           
            pred2 = clu_label2[tar_idx]
            classifier_losst2 = nn.CrossEntropyLoss()(outputs, pred2)
            classifier_losst2 *= args.cls_par
            pred3 = clu_label3[tar_idx]
            classifier_losst3 = nn.CrossEntropyLoss()(outputs, pred3)
            classifier_losst3 *= args.cls_par
            '''
        else:
            classifier_loss = torch.tensor(0.0)
            '''
            classifier_losst1 = torch.tensor(0.0).cuda()
            classifier_losst2 = torch.tensor(0.0).cuda()
            classifier_losst3 = torch.tensor(0.0).cuda()
            '''
                    
            ############################################
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
            '''
            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par
                classifier_losst1 += im_loss
                classifier_losst2 += im_loss
                classifier_losst3 += im_loss
             '''        
        
        bl_par = 0.5#(1 + 10 * iter_num / max_iter) ** (-0.75)
        #loss_tar = classifier_losst1 + classifier_losst2 + classifier_losst3 + torch.from_numpy(bl_par*(cen_loss1 + cen_loss2 + cen_loss3)).float().cuda()
        loss_tar = classifier_loss + torch.from_numpy(bl_par*(cen_loss1 + cen_loss2 + cen_loss3)).float().cuda()       
        optimizer.zero_grad()               
        loss_tar.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB1.eval()
            netC1.eval()
            netB2.eval()
            netC2.eval()
            netB3.eval()
            netC3.eval()
            for i in range(args.sk):
                netG_list[i].eval()
            #netW.eval()
            
            '''
            tar_cen1, labelset = clu_cen(dset_loaders['test'], netF, netB1, netC1, ini_cen1, args) ##pseudo            
            tar_cen2, labelset = clu_cen(dset_loaders['test'], netF, netB2, netC2, ini_cen2, args) ##pseudo            
         
            ini_cen1[labelset] = (tar_cen1[labelset] + ini_cen1[labelset])/2          
            ini_cen2[labelset] = (tar_cen2[labelset] + ini_cen2[labelset])/2
            '''
            
            ini_cen1 = clu_cen(dset_loaders['test'], netF, netB1, netC1, ini_cen1, src_rule_num1, args)           
            ini_cen2 = clu_cen(dset_loaders["test"], netF, netB2, netC2, ini_cen2, src_rule_num2, args)   
            ini_cen3 = clu_cen(dset_loaders["test"], netF, netB3, netC3, ini_cen3, src_rule_num3, args)            
          

            if args.da == 'oda':
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB1, netC1, ini_cen1, args, True, ENT_THRESHOLD1)
                log_str1 = 'Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src1, acc_os2, acc_os1, acc_unknown)
                
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB2, netC2, ini_cen2, args, True, ENT_THRESHOLD2)
                log_str2 = 'Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src2, acc_os2, acc_os1, acc_unknown)
                
                acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB3, netC3, ini_cen3, args, True, ENT_THRESHOLD3)
                log_str3 = 'Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name_src3, acc_os2, acc_os1, acc_unknown)
                
                acc_os1, acc_os2, acc_unknown= cal_globe_oda(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, ini_cen1, ini_cen2, ini_cen3, args, True, ENT_THRESHOLD1, ENT_THRESHOLD2, ENT_THRESHOLD3)                
                log_strg = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.task, iter_num, max_iter, acc_os2, acc_os1, acc_unknown)
                acc_t_te = acc_os1
            else:
                acc_st1_te, _ = cal_acc(dset_loaders['test'], netF, netB1, netC1, ini_cen1, args, False)
                log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src1, iter_num, max_iter, acc_st1_te) 
                
                acc_st2_te, _ = cal_acc(dset_loaders['test'], netF, netB2, netC2, ini_cen2, args, False)
                log_str2 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src2, iter_num, max_iter, acc_st2_te)
                
                acc_st3_te, _ = cal_acc(dset_loaders['test'], netF, netB3, netC3, ini_cen3, args, False)
                log_str3 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src3, iter_num, max_iter, acc_st3_te)
        
                
                acc_t_te, _ = cal_globe(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, netB3, netC3, netG_list, ini_cen1, ini_cen2, ini_cen3, args, False)                
                log_strg = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.task, iter_num, max_iter, acc_t_te)
                        
            args.out_file.write(log_str1 + '\n')
            args.out_file.flush()
            print(log_str1+'\n')
            args.out_file.write(log_str2 + '\n')
            args.out_file.flush()
            print(log_str2+'\n')
            args.out_file.write(log_str3 + '\n')
            args.out_file.flush()
            print(log_str3+'\n')
            args.out_file.write(log_strg + '\n')
            args.out_file.flush()
            print(log_strg+'\n')

            if acc_t_te  >= acc_init:
                acc_init = acc_t_te
                if args.issave:   
                    torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                    torch.save(netB1.state_dict(), osp.join(args.output_dir, "target_B1_" + args.savename + ".pt"))
                    torch.save(netC1.state_dict(), osp.join(args.output_dir, "target_C1_" + args.savename + ".pt"))
                    torch.save(netB2.state_dict(), osp.join(args.output_dir, "target_B2_" + args.savename + ".pt"))
                    torch.save(netC2.state_dict(), osp.join(args.output_dir, "target_C2_" + args.savename + ".pt"))
                    torch.save(netB3.state_dict(), osp.join(args.output_dir, "target_B3_" + args.savename + ".pt"))
                    torch.save(netC3.state_dict(), osp.join(args.output_dir, "target_C3_" + args.savename + ".pt"))
                    torch.save(netG_list.state_dict(), osp.join(args.output_dir, "target_W_" + args.savename + ".pt"))
                                  
            netF.train()
            netB1.train()
            netB2.train()     
            netB3.train()  
            for i in range(args.sk):
                netG_list[i].train()             
    return netF, netB1, netC1, netB2, netC2, netB3, netC3, netG_list

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s1', type=int, default=0, help="source")
    parser.add_argument('--s2', type=int, default=1, help="source")
    parser.add_argument('--s3', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='OfficeHome', choices=['VISDA-C', 'office31', 'OfficeHome', 'offcal', 'CLEF', 'DomainNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--issave', type=bool, default=False)#True
    #parser.add_argument('--rule_num', type=int, default=32)
    args = parser.parse_args()
    
    if args.dset == 'OfficeHome':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
        task = [0,1,2,3] 
        #task = [0,1,3,2]
        #task = [0,2,3,1] 
        #task = [1,2,3,0]
    '''
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'offcal':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        #task = [0,1,2,3] 
        #task = [0,1,3,2]
        #task = [0,2,3,1] 
        task = [1,2,3,0]
        args.class_num = 10
    if args.dset == 'DomainNet':
        names = ['clipart','infograph','painting', 'quickdraw', 'real', 'sketch'] 
        args.class_num = 345
        #task = [1,2,3,4,5,0] 
        task = [0,2,3,4,5,1]
        #task = [0,1,3,4,5,2] 
        #task = [0,1,2,4,5,3]
        #task = [0,1,2,3,5,4]
        #task = [0,1,2,3,4,5]    
    '''
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    '''
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    '''
    # torch.backends.cudnn.deterministic = True

    args.rule_num1 = 5 # C 5-7-7 A 6-7-7
    args.rule_num2 = 6   # P 5-6-7 R 5-6-7
    args.rule_num3 = 7 # oda A 5 C-6 P-8 R-8 
    
    args.rule_sel1 = 2
    args.rule_sel2 = 3
    args.rule_sel3 = 3
           
    args.sk = 3
    folder = './dataset/'
    args.s1_dset_path = folder + args.dset + '/' + names[task[args.s1]] + 'List.txt'
    args.s2_dset_path = folder + args.dset + '/' + names[task[args.s2]] + 'List.txt'
    args.s3_dset_path = folder + args.dset + '/' + names[task[args.s3]] + 'List.txt'
    args.test_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'  
    args.t_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'  
    
    #args.tra_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List_tra.txt' 
    #args.tst_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List_left.txt'  

    if args.dset == 'OfficeHome':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    load_tag = 'fuz_rule_0.5'#_para
    args.task = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper() + '2' + names[task[args.t]][0].upper()
    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.task, load_tag)
    traepo = 0
    save_tag = 'fuz_tar_weight' + str(traepo)
    args.output_dir = osp.join(args.output, args.da, args.dset, args.task, save_tag)

    args.name_src = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + names[task[args.s3]][0].upper()
    args.name_src1 = names[task[args.s1]][0].upper()
    args.name_src2 = names[task[args.s2]][0].upper()
    args.name_src3 = names[task[args.s3]][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)
