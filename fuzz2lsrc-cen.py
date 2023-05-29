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

def data_load_src(args, source_name): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(source_name).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()
        
    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList_idx(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList_idx(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

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

def data_load_src_te(args, source_name): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(source_name).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

    dsets["source"] = ImageList_idx(txt_src, transform=image_test())#te_txt
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def srcnet_output(inputs, netF, netB, netC, cen, args):
    
    feas = netB(netF(inputs))
    mem_ship = clu_mem(feas.detach().cpu(), cen, args)
    tar_rule = np.argsort(-mem_ship, axis=1)
    mem_ship = torch.from_numpy(mem_ship).float().cuda()            
    outputs = netC(feas) 
        
    return mem_ship, tar_rule, outputs
    
def cal_globe(loader, netF, netB1, netC1, netB2, netC2, cen1, cen2, args, flag=False):
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
           
            src_rule_num = int(args.rule_num1/2)
            outputs1 = cal_output_sel(outputs1, mem_ship1, tar_rule1, src_rule_num, args)
            src_rule_num = int(args.rule_num2/2)
            outputs2 = cal_output_sel(outputs2, mem_ship2, tar_rule2, src_rule_num, args)
            
            outputs = (outputs1+outputs2)/2
            
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
    #loader = dset_loaders_s1['source_te']
    #netB = netB1 netC = netC1 cen = cen_source1
    test_loss = 0
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
            ##############################
            output_loss = torch.nn.functional.softmax(outputs, dim=1)
            test_loss += F.nll_loss(F.log_softmax(output_loss, dim=1), labels.cuda()).item()
            
                     
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
    #####################################################
    #test_loss = F.nll_loss(F.log_softmax(all_output, dim=1), all_label).item()
    test_loss /= len(all_label)
    ########################################################
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent, test_loss

def clu_cen(loader, netF, netB, src_path, src_rule, args):
    start_test = True 
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            #labels = data[1] # ground truth
            #idx = data[2]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))

            if start_test:
                all_fea = feas.float().cpu()
                #all_idx = idx.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                #all_idx = torch.cat((all_idx, idx.float()), 0)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()  
    #all_idx = all_idx.int().cpu().numpy()   
    
    K = src_rule#all_output.size(1)
    all_label = np.load(src_path)
    #all_label = all_label[all_idx]
    aff = np.eye(K)[all_label] #label vector sample size classes  .int()
        
    cls_count = np.eye(K)[all_label].sum(axis=0) #cluster number 
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    initc = update_cen_src(aff, all_fea, labelset, K, args)
    
    return initc

def update_cen_src(aff, all_fea, labelset, K, args):
    
    initc0 = aff.transpose().dot(all_fea)
    initc0 = initc0 / (1e-8 + aff.sum(axis=0)[:,None])
        
    '''
    initc0=np.random.rand(5,3)
    all_fea=np.random.rand(10,3)
    dist_c = cdist(all_fea, initc0, 'cosine')#
    dda=dist_a.repeat(5, axis=1)
    a=dist_c*dda
    initc2 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])
    '''
    
    dist_c = cdist(all_fea, initc0[labelset], args.distance)#
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(len(labelset), axis=1)
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy() 
    #np.savetxt(args.output_dir_src+"/"+args.name_src1+"_mem_ship.csv", mem_ship, fmt='%.4f', delimiter=',')
    #nn.Softmax(dim=1)().numpy(torch.from_numpy())

    for round in range(1):
        aff = np.power(mem_ship,2)#mem_ship #########################
        initc1 = aff.transpose().dot(all_fea)
        initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])   
    
    return initc1

def update_anc_src(aff, all_fea, labelset, K, args):
    
    initc0 = aff.transpose().dot(all_fea)
    initc0 = initc0 / (1e-8 + aff.sum(axis=0)[:,None])  
    
    return initc0

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

def rule_loss(output, labels, src_rule_num):
    loss = torch.tensor(0.0).cuda()
    for i in range(src_rule_num):
        loss += CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(output[i], labels)
   
    return loss

def cal_loss(inputs, labels, netF, netB, netC, cen, src_rule_num, args):
    
    mem_ship, tar_rule, output = srcnet_output(inputs, netF, netB, netC, cen, args)                                                                    			
    loss = rule_loss(output, labels, src_rule_num)
    output = cal_output(output, mem_ship, args)
    loss += CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(output, labels)
        
    return loss

def cal_loss_aux(inputs, labels, netF, netB, netC, cen, src_rule_num, args):
    
    mem_ship, tar_rule, output = srcnet_output(inputs, netF, netB, netC, cen, args)                                   
                  			
    output = cal_output_sel(output, mem_ship, tar_rule, src_rule_num, args)
    loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(output, labels)
        
    return loss

def anc_tra(inputs, labels, netF, netB, netC, cen, args):    
    
    mem_ship, tar_rule, output = srcnet_output(inputs, netF, netB, netC, cen, args) 
    output = cal_output(output, mem_ship, args)
    output = nn.Softmax(dim=1)(output)
    
    fea = netB(netF(inputs)).detach().cpu()
    if args.distance == 'cosine':
        fea = norm_fea(fea).float().numpy()
    
    aff = output.detach().float().cpu().numpy()
    initc = aff.transpose().dot(fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])   
    return initc

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


def train_source(args):
    dset_loaders_s1 = data_load_src(args, args.s1_dset_path)
    dset_loaders_s2 = data_load_src(args, args.s2_dset_path)
    dset_loaders = data_load_tst(args)   
    
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB1 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num = args.rule_num1).cuda()
    netB2 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num = args.rule_num2).cuda()       
    
    '''
    src_anc1 = clu_anc_ini(dset_loaders_s1["source_tr"], netF, netB1, args)
    src_anc2 = clu_anc_ini(dset_loaders_s2["source_tr"], netF, netB2, args)
    
    src_anc1 = torch.tensor(src_anc1).cuda()
    src_anc2 = torch.tensor(src_anc2).cuda()
    '''
    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC1.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    for k, v in netB2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC2.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}] 

    #param_group += [{'params': src_anc1, 'lr': learning_rate}]
    #param_group += [{'params': src_anc2, 'lr': learning_rate}]
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_epo = max(len(dset_loaders_s1["source_tr"]),len(dset_loaders_s2["source_tr"]))#, len(dset_loaders["target"])
    
    max_iter = args.max_epoch * max_epo
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB1.train()
    netC1.train()
    netB2.train()
    netC2.train()
    loss_tra_tst = []


    while iter_num < max_iter:
        try:
            inputs_source1, labels_source1, s1_idx = iter_source1.next()
        except:
            iter_source1 = iter(dset_loaders_s1["source_tr"])
            inputs_source1, labels_source1, s1_idx = iter_source1.next()

        try:
            inputs_source2, labels_source2, s2_idx = iter_source2.next()
        except:
            iter_source2 = iter(dset_loaders_s2["source_tr"])
            inputs_source2, labels_source2, s2_idx = iter_source2.next()

        if inputs_source1.size(0) == 1 or inputs_source2.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB1.eval()
            netB2.eval()
            
            src_path = args.src_path1
            src_rule = args.rule_num1
            src_loaders = data_load_src_te(args, args.s1_dset_path)  
            cen_source1 = clu_cen(src_loaders["source"], netF, netB1, src_path, src_rule, args)         
            
            src_path = args.src_path2
            src_rule = args.rule_num2
            src_loaders = data_load_src_te(args, args.s2_dset_path)
            cen_source2 = clu_cen(src_loaders["source"], netF, netB2, src_path, src_rule, args) 
            del src_loaders
            
            netF.train()
            netB1.train()
            netB2.train()
                   
        
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source1, labels_source1 = inputs_source1.cuda(), labels_source1.cuda()
        inputs_source2, labels_source2 = inputs_source2.cuda(), labels_source2.cuda()
    
           
        src_rule_num = args.rule_num1
        classifier_loss1 = cal_loss(inputs_source1, labels_source1, netF, netB1, netC1, cen_source1, src_rule_num, args)                       
        src_rule_num = int(args.rule_num2/2)
        cro_loss1 = cal_loss_aux(inputs_source1, labels_source1, netF, netB2, netC2, cen_source2, src_rule_num, args)                       
        
        '''
        anc1 = anc_tra(inputs_source1, labels_source1, netF, netB1, netC1, cen_source1, args)
        cen_loss1 = np.linalg.norm(src_anc1.detach().cpu().float().numpy() - anc1, ord=2, keepdims=True)         
        '''
        loss1 = classifier_loss1 + cro_loss1 #+ torch.from_numpy(cen_loss1).float().cuda()
        
        optimizer.zero_grad()               
        loss1.backward()
        optimizer.step()       
        
        src_rule_num = args.rule_num2
        classifier_loss2 = cal_loss(inputs_source2, labels_source2, netF, netB2, netC2, cen_source2, src_rule_num, args)                       
        src_rule_num = int(args.rule_num1/2)
        cro_loss2 = cal_loss_aux(inputs_source2, labels_source2, netF, netB1, netC1, cen_source1, src_rule_num, args)                               
        '''
        anc2 = anc_tra(inputs_source2, labels_source2, netF, netB2, netC2, cen_source2, args)
        cen_loss2 = np.linalg.norm(src_anc2.detach().cpu().float().numpy() - anc2, ord=2, keepdims=True)         
        '''
        loss2 = classifier_loss2 + cro_loss2 #+ torch.from_numpy(cen_loss2).float().cuda()
        
        optimizer.zero_grad()               
        loss2.backward()
        optimizer.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB1.eval()
            netC1.eval()
            netB2.eval()
            netC2.eval()

            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB1, netC1, cen_source1, args, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s1_te, _, test_loss1 = cal_acc(dset_loaders_s1['source_te'], netF, netB1, netC1, cen_source1, args, False)
                acc_st1_te, _, test_loss1t = cal_acc(dset_loaders['test'], netF, netB1, netC1, cen_source1, args, False)
                log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Accuracy_st1 = {:.2f}%'.format(args.name_src1, iter_num, max_iter, acc_s1_te, acc_st1_te) 
                
                acc_s2_te, _, test_loss2 = cal_acc(dset_loaders_s2['source_te'], netF, netB2, netC2, cen_source2, args, False)                
                acc_st2_te, _, test_loss2t = cal_acc(dset_loaders['test'], netF, netB2, netC2, cen_source2, args, False)
                log_str2 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Accuracy_st2 = {:.2f}%'.format(args.name_src2, iter_num, max_iter, acc_s2_te, acc_st2_te)

                acc_t_te, _ = cal_globe(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, cen_source1, cen_source2, args, False)                
                log_strg = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.task, iter_num, max_iter, acc_t_te)
                
                loss_tra_tst.append([classifier_loss1.item(), classifier_loss2.item(), test_loss1, test_loss1t, test_loss2, test_loss2t, acc_s1_te, acc_st1_te, acc_s2_te, acc_st2_te])
                np.savetxt('{}/train-test-loss.csv'.format(args.output_dir_src), np.array(loss_tra_tst), fmt='%.6f', delimiter=',')
            
            
            args.out_file.write(log_str1 + '\n')
            args.out_file.flush()
            print(log_str1+'\n')
            args.out_file.write(log_str2 + '\n')
            args.out_file.flush()
            print(log_str2+'\n')
            args.out_file.write(log_strg + '\n')
            args.out_file.flush()
            print(log_strg+'\n')

            if (acc_s1_te + acc_s2_te)  >= acc_init:
                acc_init = acc_s1_te + acc_s2_te
                best_netF = netF.state_dict()
                best_netB1 = netB1.state_dict()
                best_netC1 = netC1.state_dict()
                best_netB2 = netB2.state_dict()
                best_netC2 = netC2.state_dict()
                
                #best_anc1 = src_anc1.detach().cpu().float().numpy()
                #best_anc2 = src_anc2.detach().cpu().float().numpy()

                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB1, osp.join(args.output_dir_src, args.name_src1+"_B.pt"))
                torch.save(best_netC1, osp.join(args.output_dir_src, args.name_src1+"_C.pt"))
                torch.save(best_netB2, osp.join(args.output_dir_src, args.name_src2+"_B.pt"))
                torch.save(best_netC2, osp.join(args.output_dir_src, args.name_src2+"_C.pt"))
                #np.save(args.output_dir_src+"/"+args.name_src1+"_anc.npy", best_anc1)
                #np.save(args.output_dir_src+"/"+args.name_src2+"_anc.npy", best_anc2)
                cal_src_cen(args)


            netF.train()
            netB1.train()
            netC1.train()
            netB2.train()
            netC2.train()
                   
    return netF, netB1, netC1, netB2, netC2

def clu_cen_ini(loader, netF, netB, src_path, src_rule, args):
    start_test = True #loader = dset_loaders_s1["source_te"]
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            #labels = data[1] # ground truth
            inputs = inputs.cuda()
            feas = netB(netF(inputs))

            if start_test:
                all_fea = feas.float().cpu()
                #all_label = labels.float()
                #all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                #all_label = torch.cat((all_label, labels.float()), 0)
                #all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    
    K = src_rule#args.class_num#all_output.size(1)
    all_label = np.load(src_path)
    aff = np.eye(K)[all_label] #label vector sample size classes  .int()
        
    cls_count = np.eye(K)[all_label].sum(axis=0) #cluster number 
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    initc = update_cen_src(aff, all_fea, labelset, K, args)

    return initc


def clu_anc_ini(loader, netF, netB, args):
    start_test = True #loader = dset_loaders_s1["source"]
    with torch.no_grad(): # netB = netB1
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1] # ground truth
            inputs = inputs.cuda()
            feas = netB(netF(inputs))

            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                #all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                #all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    
    K = args.class_num#all_output.size(1)
    aff = np.eye(K)[all_label.int()] #label vector sample size classes
        
    cls_count = np.eye(K)[all_label.int()].sum(axis=0) #cluster number 
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    initc = update_anc_src(aff, all_fea, labelset, K, args)

    return initc

def clu_anc_para(netC, args):#netC=netC1
    para=[]
    for k,v in netC.named_parameters():
        v.requires_grad = False
        last=k.rfind('v')
        if k[last] =='v':
            #print(v.shape)
            para.append(v.cpu().detach())
    
    '''
    cen = np.load(args.output_dir_src+"/"+args.name_src1+"_cen.npy")
    mem_ship = clu_mem(para[0], cen, args)
    tar_rule = np.argsort(-mem_ship, axis=1)
    mem_ship = torch.from_numpy(mem_ship).float().cuda()            
    outputs = netC(para[0].cuda()) 
    outputs = cal_output(outputs, mem_ship, args) #predicted
    output = nn.Softmax(dim=1)(outputs)
    _, predict = torch.max(output, 1)
    '''
                
    anc = np.zeros([para[0].shape[0], para[0].shape[1]+1])
    for i in range(len(para)):#i=0
        parai= para[i]
        parai = norm_fea(parai).numpy()
        anc = anc + parai
    anc = anc / len(para) 
    return anc

def cal_src_cen(args):
    dset_loaders_s1 = data_load_src_te(args, args.s1_dset_path)
    dset_loaders_s2 = data_load_src_te(args, args.s2_dset_path)
    dset_loaders = data_load_tst(args)
       
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB1 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num=args.rule_num1).cuda()
    netB2 = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifierf(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck, rule_num=args.rule_num2).cuda()
        
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


    netF.eval()
    netB1.eval()
    netC1.eval()
    netB2.eval()
    netC2.eval()
    
    src_path = args.src_path1
    src_rule = args.rule_num1
    src_cen1 = clu_cen_ini(dset_loaders_s1["source"], netF, netB1, src_path, src_rule, args)
    #src_anc1 = clu_anc_ini(dset_loaders_s1["source"], netF, netB1, args)
    src_anc1 = clu_anc_para(netC1, args)
    
    
    src_path = args.src_path2
    src_rule = args.rule_num2
    src_cen2 = clu_cen_ini(dset_loaders_s2["source"], netF, netB2, src_path, src_rule, args)
    #src_anc2 = clu_anc_ini(dset_loaders_s2["source"], netF, netB2, args)
    src_anc2 = clu_anc_para(netC2, args)

    
    ini_cen1 = src_cen1 
    ini_cen2 = src_cen2
    
    np.save(args.output_dir_src+"/"+args.name_src1+"_cen.npy", src_cen1)
    np.save(args.output_dir_src+"/"+args.name_src2+"_cen.npy", src_cen2)
   
    np.save(args.output_dir_src+"/"+args.name_src1+"_anc_para.npy", src_anc1)
    np.save(args.output_dir_src+"/"+args.name_src2+"_anc_para.npy", src_anc2)

    '''
    if args.dset=='VISDA-C':
        acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB1, netC1, ini_cen1, args, True)
        log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.name_src, acc_s_te) + '\n' + acc_list
    else:
        acc_st1_te, _ = cal_acc(dset_loaders['test'], netF, netB1, netC1, ini_cen1, args, False)
        log_str1 = 'Task: {}, Accuracy = {:.2f}%'.format(args.name_src1, acc_st1_te) 
        
        acc_st2_te, _ = cal_acc(dset_loaders['test'], netF, netB2, netC2, ini_cen2, args, False)
        log_str2 = 'Task: {}, Accuracy = {:.2f}%'.format(args.name_src2, acc_st2_te)

        acc_t_te, _ = cal_globe(dset_loaders['test'], netF, netB1, netC1, netB2, netC2, ini_cen1, ini_cen2, args, False)                
        log_strg = 'Task: {}, Accuracy = {:.2f}%'.format(args.task, acc_t_te)           

    args.out_file.write(log_str1 + '\n')
    args.out_file.flush()
    print(log_str1+'\n')
    args.out_file.write(log_str2 + '\n')
    args.out_file.flush()
    print(log_str2+'\n')
    args.out_file.write(log_strg + '\n')
    args.out_file.flush()
    print(log_strg+'\n')
    '''



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
    parser.add_argument('--t', type=int, default=2, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office31', choices=['VISDA-C', 'office31', 'OfficeHome', 'offcal', 'CLEF', 'DomainNet'])
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
    parser.add_argument('--issave', type=bool, default=True)
    #parser.add_argument('--rule_num', type=int, default=5)
    args = parser.parse_args()
    
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        #task = [0,1,2] 
        task = [2,1,0]
        #task = [0,2,1]

    if args.dset == 'CLEF':
        names = ['p','c','i']
        args.class_num = 12
        #pc-i,ic-p, ip-c: 
        task = [0,1,2] 
        #task = [2,1,0]
        #task = [2,0,1] 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    '''
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    '''
    # torch.backends.cudnn.deterministic = True

    args.rule_num1 = 15  # A 15-16 D 15-15
    args.rule_num2 = 16   # W 15-16 
    
    args.sk = 2
    folder = './dataset/'
    args.s1_dset_path = folder + args.dset + '/' + names[task[args.s1]] + 'List.txt'
    args.s2_dset_path = folder + args.dset + '/' + names[task[args.s2]] + 'List.txt'
    args.test_dset_path = folder + args.dset + '/' + names[task[args.t]] + 'List.txt'     

    if args.dset == 'OfficeHome':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    traepo = 'para_tra_tst_loss'#'fuz_rule_cmean'#fuz_rule_0.5_para
    args.task = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper() + '2' + names[task[args.t]][0].upper()
    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.task, traepo)
    args.name_src = names[task[args.s1]][0].upper() + names[task[args.s2]][0].upper()
    args.name_src1 = names[task[args.s1]][0].upper()
    args.name_src2 = names[task[args.s2]][0].upper()
    
    args.src_path1 = args.output_dir_src+"/"+args.name_src1+"_clu_lab.npy"
    args.src_path2 = args.output_dir_src+"/"+args.name_src2+"_clu_lab.npy"

    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    
    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)    
    #cal_src_cen(args)
