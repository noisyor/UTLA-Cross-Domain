#!/usr/bin/env python
# coding: utf-8

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import os
import math
from scipy.stats import norm
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  
from sklearn import preprocessing
import itertools
import random
import sys
import params
from sklearn.decomposition import PCA

# main
set_UTLA_train= int(sys.argv[1])
source_device= int(sys.argv[2]) 
target_device= int(sys.argv[3]) 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

### handle the dataset
class TorchDataset(Dataset):
    def __init__(self, trs_file, label_file, trace_num, trace_offset, trace_length):
        self.trs_file = trs_file
        self.label_file = label_file
        self.trace_num = trace_num
        self.trace_offset = trace_offset
        self.trace_length = trace_length
        self.ToTensor = transforms.ToTensor()
    def __getitem__(self, i):
        index = i % self.trace_num
        trace = self.trs_file[index,:]
        label = self.label_file[index]
        trace = trace[self.trace_offset:self.trace_offset+self.trace_length]
        trace = np.reshape(trace,(1,-1))
        trace = self.ToTensor(trace)
        trace = np.reshape(trace, (1,-1))
        label = torch.tensor(label, dtype=torch.long) 
        return trace.float(), label, index
    def __len__(self):
        return self.trace_num
    
### data loader for training
def load_training(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)
    return train_loader

### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    return test_loader


Sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71,
        240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216,
        49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160,
        82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208,
        239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
        182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96,
        129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211,
        172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
        120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97,
        53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140,
        161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]

HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
            3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
            4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
            6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
            4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
            6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

InvSbox = [82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255, 135,
           52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61,238, 76, 149, 11, 66, 250, 195, 78, 8,
           46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22, 212,
           164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132, 144,
           216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189,
           3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172, 116, 34,
           231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170,
           24,190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51, 136, 7,
           199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181,74, 13, 45, 229, 122, 159, 147, 201, 156,
           239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214, 38,
           225, 105, 20, 99, 85, 33,12, 125]

### test/attack
def test(model, profile_model, device_id, model_flag='pretrained'):
    """
    - model : the current model 
    - device_id : id of the tested device
    - disp_GE : whether to attack/calculate guessing entropy (GE)
    - model_flag : a string for naming GE result
    """
    # enter evaluation mode
    model.eval()
    test_loss = 0
    # the number of correct prediction
    correct = 0
    test_loss = 0 
    mmd_loss = 0
    encoder_loss = 0
    clf_criterion = nn.CrossEntropyLoss()
    test_num = target_test_num
    test_loader = target_test_loader
    real_key = real_key_02
    test_preds_all = torch.zeros((test_num, class_num), dtype=torch.float, device='cpu')
    iter_target = iter(target_test_loader)
    iter_source = iter(source_test_loader)
    iter_num = len(target_test_loader)
    for i in range(0,iter_num):
        source_data, _ , _ = next(iter_source)
        target_data, target_label, _ = next(iter_target)
        if cuda:
            source_data = source_data.cuda()
            target_data = target_data.cuda()
            target_label = target_label.cuda()
        target_data = Variable(target_data)
        source_data = Variable(source_data)
        target_label = Variable(target_label)
        target_preds, target_feat, target_featm1, target_featm2 = model(target_data)
        _, source_feat, source_featm1, source_featm2 = profile_model(source_data)
        softmax = nn.Softmax(dim=1)  # softmax for the output layer
        test_preds_all[i*batch_size:(i+1)*batch_size, :] = softmax(target_preds)
        test_loss += clf_criterion(target_preds, target_label)

        mmd_loss1 = mmd_rbf(source_feat, target_feat)  # MMD loss between source and target features
        mmd_loss2 = mmd_rbf(source_featm1, target_featm1) # MMD loss between previous features (for stability)
        mmd_loss3 = mmd_rbf(source_featm2, target_featm2)

        mmd_loss += lambda1*mmd_loss1 + lambda2*mmd_loss2 + lambda3*mmd_loss3 # total MMD loss, lambda1, lambda2, lambda3 are penalty coefficients
        encoder_loss += clf_criterion(target_feat, Variable((torch.ones(target_feat.size(0)).long()).cuda()))
        pred = target_preds.data.max(1)[1]
        correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader)
    mmd_loss /= len(test_loader)
    encoder_loss /= len(test_loader)
    print('Target test loss: {:.4f}, Target test accuracy: {}/{} ({:.2f}%), MMD loss: {:.4f}, Encoder Loss: {:.4f}\n'.format(
        test_loss.data, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), mmd_loss.data, encoder_loss.data))
    plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)

def encoder_rep(epoch, model, profile_model):
    # enter evaluation mode
    model.eval()
    profile_model.eval()
    test_num = target_test_num
    target_feat_preds = torch.zeros((test_num, 192), dtype=torch.float, device='cpu')
    source_feat_preds = torch.zeros((test_num, 192), dtype=torch.float, device='cpu')
    iter_target = iter(target_test_loader)
    iter_source = iter(source_test_loader)
    iter_num = len(target_test_loader)
    for i in range(0,iter_num):
        source_data, _ , _ = next(iter_source)
        target_data, target_label, _ = next(iter_target)
        if cuda:
            source_data = source_data.cuda()
            target_data = target_data.cuda()
            target_label = target_label.cuda()
        target_data = Variable(target_data)
        source_data = Variable(source_data)
        target_label = Variable(target_label)
        _, target_feat, _, _ = model(target_data)
        _, source_feat, _, _ = profile_model(source_data)
        target_feat_preds[i*batch_size:(i+1)*batch_size, :] = target_feat.cpu()  # save the target features
        source_feat_preds[i*batch_size:(i+1)*batch_size, :] = source_feat.cpu()  # save the source features
    
    # Convert tensors to NumPy arrays
    target_feat_preds_np = target_feat_preds.detach().numpy()
    source_feat_preds_np = source_feat_preds.detach().numpy()

    # Concatenate the target and source feature predictions
    combined_feats = np.concatenate((target_feat_preds_np, source_feat_preds_np), axis=0)

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_feats)

    # Split the PCA results back into target and source
    target_pca = pca_result[:target_feat_preds_np.shape[0], :]
    source_pca = pca_result[target_feat_preds_np.shape[0]:, :]
    # Calculate the range of the source features
    x_min, x_max = source_pca[:, 0].min(), source_pca[:, 0].max()
    y_min, y_max = source_pca[:, 1].min(), source_pca[:, 1].max()

    # Add some padding to the limits for better visualization
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    x_min -= 3*x_padding
    x_max += 3*x_padding
    y_min -= 3*y_padding
    y_max += 3*y_padding

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(source_pca[:, 0], source_pca[:, 1], label='Source Features', alpha=0.5, color='red', marker='o')
    plt.scatter(target_pca[:, 0], target_pca[:, 1], label='Target Features', alpha=0.5, color='blue', marker='d')
    # Set fixed axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('PCA Component 1', fontsize=14)  # x-axis label
    plt.ylabel('PCA Component 2', fontsize=14)  # y-axis label
    plt.title('Epoch = %d ' % (epoch), fontsize=16)
    plt.legend()
    plt.tight_layout()  # Automatically adjust padding to prevent clipping  
    plt.show()
    plt.savefig('./figures/PCA_UTLA_crossSC_{}_to_{}_'.format(source_device_id, target_device_id) + '_epoch_{}'.format(epoch)+'.png')

def test_intermediate(model):
    # enter evaluation mode
    model.eval()
    real_key = real_key_02
    # Initialize the prediction and label lists(tensors)
    iter_target = iter(target_finetune_loader)
    # max trace num for attack
    trace_num_max = 1000
    num_iter = int(trace_num_max/batch_size)
    guessing_entropy = np.zeros(trace_num_max)
    mean_est = np.zeros([256])
    var_est = np.zeros([256])
    guessing_entropy_neg = np.zeros(trace_num_max)
    mean_est_neg = np.zeros([256])
    var_est_neg = np.zeros([256])
    score_mat = np.zeros((trace_num_max, 256))
    plaintext = plaintexts_target_train
    test_preds_all = torch.zeros((trace_num_max, class_num), dtype=torch.float, device='cpu')
    ordering = np.zeros(trace_num_max)
    label_ordering = np.zeros(trace_num_max)

    for i in range(0,num_iter):
        target_data, target_label, target_idx = next(iter_target)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data, target_label = Variable(target_data), Variable(target_label)
        target_preds, _, _, _ = model(target_data)
        softmax = nn.Softmax(dim=1)  # Use softmax to get probabilities
        test_preds_all[i*batch_size:(i+1)*batch_size, :] = softmax(target_preds)
        ordering[i*batch_size:(i+1)*batch_size] = target_idx
        label_ordering[i*batch_size:(i+1)*batch_size] = target_label.cpu()

    #breakpoint()
    for i in range(0,trace_num_max):
        trueState = plaintext[int(ordering[i])] ^ real_key
        truelabel = Sbox[trueState]
        #breakpoint()
        for key_guess in range(0, 256):
            initialState = plaintext[int(ordering[i])] ^ key_guess
            label = Sbox[initialState]
            score_mat[i, key_guess] = test_preds_all[i, label] - test_preds_all[i, truelabel]
    
    score_mat_neg = np.copy(score_mat)*-1

    for key_guess in range(0, 256):
        mean_est[key_guess] = np.mean(score_mat[:,key_guess])
        var_est[key_guess] = np.var(score_mat[:,key_guess])
        mean_est_neg[key_guess] = np.mean(score_mat_neg[:,key_guess])
        var_est_neg[key_guess] = np.var(score_mat_neg[:,key_guess])

    # attack multiples times for average
    for i in range(0,trace_num_max):
        guessing_entropy[i] = 1
        guessing_entropy_neg[i] = 1
        for fk in range(0, 256):
            if(fk!=real_key):
                guessing_entropy[i] = guessing_entropy[i]  + norm.cdf(np.sqrt(i+1)*mean_est[fk]/np.sqrt(var_est[fk]), loc=0, scale=1)
                guessing_entropy_neg[i] = guessing_entropy_neg[i]  + norm.cdf(np.sqrt(i+1)*mean_est_neg[fk]/np.sqrt(var_est_neg[fk]), loc=0, scale=1)

    if(guessing_entropy_neg[-1] < guessing_entropy[-1]):
        # if negative score is better, use negative score
        guessing_entropy = guessing_entropy_neg

    guessing_entropy_int = guessing_entropy.astype(int)
    if(np.size(np.where(guessing_entropy_int<2))==0):
        output_str = trace_num_max-1
    else:
        # find the first point where GE < 2
        output_str = np.where(guessing_entropy_int<2)[0][0]
    #breakpoint()
    return output_str+1,guessing_entropy[-1]

### UTLA
def UTLA_train(epoch, atn_target, atn_profile, critic):
    """
    - epoch : the current epoch
    - atn_target: the adversarial transfer network for the target device
    - atn_profile: the adversarial transfer network for the profiling device
    - critic: the Discriminator
    """
    
    # enter training mode
    atn_target.train()
    critic.train()
    atn_profile.eval()
    # get the number of batches
    clf_criterion = nn.CrossEntropyLoss()
    # train on each batch of data
    #for i in range(1, num_iter_source+1):
    data_zip = enumerate(zip(source_train_loader, target_finetune_loader))
    for step, ((images_src, _, _), (images_tgt, _, _)) in data_zip:
        # get traces and labels for source domain
        source_data = make_variable(images_src)
        target_data = make_variable(images_tgt)
        ############################
        ### Train  Discriminator ###
        ############################
        optimizer_critic.zero_grad()

        # extract and concat features
        _,feat_s,_, _ = atn_profile(source_data)
        _,feat_t,_, _ = atn_target(target_data)
        feat_concat = torch.cat((feat_s, feat_t), 0)

        # predict on discriminator
        pred_concat = critic(feat_concat.detach())

        # prepare domain labels, 1 for source device, 0 for target device
        critic_label_s = make_variable((torch.ones(feat_s.size(0)).long()).cuda())
        critic_label_t = make_variable((torch.zeros(feat_t.size(0)).long()).cuda())
        critic_label_concat = torch.cat((critic_label_s, critic_label_t), 0)
        
        # compute loss for critic
        loss_critic = clf_criterion(pred_concat, critic_label_concat)
        loss_critic.backward()

        # optimize critic
        optimizer_critic.step()
        
        ############################
        ### Train   the  Encoder ###
        ############################
        # zero gradients for optimizer
        optimizer_model.zero_grad()

        # extract target features
        _,feat_s,feat_sm1,feat_sm2 = atn_profile(source_data)
        _,feat_t,feat_tm1,feat_tm2 = atn_target(target_data)

        # predict on discriminator
        pred_t_enctrain = critic(feat_t.detach())

        # prepare fake labels
        fake_label_t = make_variable((torch.ones(pred_t_enctrain.size(0)).long()).cuda())

        # compute adversarial discriminator loss 
        loss_tgt = clf_criterion(pred_t_enctrain, fake_label_t)

        # compute mmd-loss
        mmd_loss1 = mmd_rbf(feat_s,feat_t)
        mmd_loss2 = mmd_rbf(feat_sm1,feat_tm1)
        mmd_loss3 = mmd_rbf(feat_sm2,feat_tm2)

        # compute classification loss on source data
        loss_mmd = lambda1*mmd_loss1 + lambda2*mmd_loss2 + lambda3*mmd_loss3 

        total_loss = loss_tgt + loss_mmd
        
        #total_loss.backward()
        total_loss.backward()

        # optimize total loss
        optimizer_model.step()

        # predict on discriminator
        _,feat_t_enctrain,_, _ = atn_target(target_data)
        feat_concat_enctrain = torch.cat((feat_s, feat_t_enctrain), 0)
        pred_concat_enctrain = critic(feat_concat_enctrain.detach())
        loss_critic_enctrain = clf_criterion(pred_concat_enctrain, critic_label_concat)
        preds_encoder = pred_concat_enctrain.data.max(1, keepdim=True)[1]
        
        # get the number of correct prediction
        correct_batch_encoder = preds_encoder.eq(critic_label_concat.data.view_as(preds_encoder)).float().mean()
        #breakpoint()
        if step % log_interval == 0:
            #print(torch.eq(preds_encoder,preds_disc_train).all())
            print('Epoch Encoder {}: [{}/{} ({:.0f}%)]\tcritic_loss: {:.2f}\tencoder_loss: {:.2f}\tcritic_acc: {:.2f}'.format(
                epoch, step * len(source_data), len(source_train_loader) * batch_size, 100. * step / len(source_train_loader), loss_critic_enctrain.data,
                loss_tgt.data, correct_batch_encoder * 100))
            

### show the guessing entropy 
def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    """
    - preds : the probability for each class (n*256 for a byte)
    - real_key : the key of the target device
    - device_id : id of the target device
    - model_flag : a string for naming GE result
    """
    # max trace num for attack
    trace_num_max = 1000
    guessing_entropy = np.zeros(trace_num_max)
    guessing_entropy_neg = np.zeros(trace_num_max)
    mean_est = np.zeros([256])
    var_est = np.zeros([256])
    mean_est_neg = np.zeros([256])
    var_est_neg = np.zeros([256])
    score_mat = np.zeros((trace_num_max, 256))
    plaintext = plaintexts_target_attack

    for i in range(0,trace_num_max):
        for key_guess in range(0, 256):
            initialState = plaintext[i] ^ key_guess
            label = Sbox[initialState]
            trueState = plaintext[i] ^ real_key
            truelabel = Sbox[trueState]
            score_mat[i, key_guess] = preds[i, label] - preds[i, truelabel]
    
    score_mat_neg = np.copy(score_mat)*-1

    for key_guess in range(0, 256):
        mean_est[key_guess] = np.mean(score_mat[:,key_guess])
        var_est[key_guess] = np.var(score_mat[:,key_guess])
        mean_est_neg[key_guess] = np.mean(score_mat_neg[:,key_guess])
        var_est_neg[key_guess] = np.var(score_mat_neg[:,key_guess])

    # attack multiples times for average
    for i in range(0,trace_num_max):
        guessing_entropy[i] = 1
        guessing_entropy_neg[i] = 1
        for fk in range(0, 256):
            if(fk!=real_key):
                guessing_entropy[i] = guessing_entropy[i]  + norm.cdf(np.sqrt(i+1)*mean_est[fk]/np.sqrt(var_est[fk]), loc=0, scale=1)
                guessing_entropy_neg[i] = guessing_entropy_neg[i]  + norm.cdf(np.sqrt(i+1)*mean_est_neg[fk]/np.sqrt(var_est_neg[fk]), loc=0, scale=1)

    if(guessing_entropy_neg[-1] < guessing_entropy[-1]):
        # if negative score is better, use negative score
        guessing_entropy = guessing_entropy_neg


    guessing_entropy_int = guessing_entropy.astype(int)
    if(np.size(np.where(guessing_entropy_int<2))==0):
        output_str = trace_num_max-1 
    else:
        # find the first point where GE < 2
        output_str = np.where(guessing_entropy_int<2)[0][0]
    
    print(output_str+1, guessing_entropy[-1]) 
    plt.figure(figsize=(6,4))
    plt.plot(guessing_entropy,color='red')
    plt.xlabel('Number of traces', fontsize=14)  # x-axis label
    plt.ylabel('Guessing entropy', fontsize=14)  # y-axis label
    ax = plt.gca()  # get current axis
    ax.tick_params(axis='y', labelsize=12) 
    ax.tick_params(axis='x', labelsize=12)   
    plt.ylim((0, 256))
    plt.tight_layout()  # Automatically adjust padding to prevent clipping  
    plt.show()
    plt.savefig('./figures/GE_UTLA_crossSC_XMEGA_{}_to_{}_'.format(source_device_id, device_id) + model_flag + '.png') 
    np.save('./results/GE_UTLA_crossSC_XMEGA_{}_to_{}_'.format(source_device_id, device_id) + model_flag, guessing_entropy)

### kernel function
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    - source : source data
    - target : target data
    - kernel_mul : multiplicative step of bandwidth (sigma)
    - kernel_num : the number of guassian kernels
    - fix_sigma : use a fix value of bandwidth
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    # |x-y|
    L2_distance = ((total0-total1)**2).sum(2) 
    
    # bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # take the current bandwidth as the median value, and get a list of bandwidths (for example, when bandwidth is 1, we get [0.25,0.5,1,2,4]). 
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # exp(-|x-y|/bandwidth)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    # return the final kernel matrix
    return sum(kernel_val)

### MMD loss function based on guassian kernels
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    - source : source data
    - target : target data
    - kernel_mul : multiplicative step of bandwidth (sigma)
    - kernel_num : the number of guassian kernels
    - fix_sigma : use a fix value of bandwidth
    """
    loss = 0.0
    batch_mmd_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,kernel_mul=kernel_mul,kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_mmd_size, :batch_mmd_size] # Source<->Source
    YY = kernels[batch_mmd_size:, batch_mmd_size:] # Target<->Target
    XY = kernels[:batch_mmd_size, batch_mmd_size:] # Source<->Target
    YX = kernels[batch_mmd_size:, :batch_mmd_size] # Target<->Source
    loss = torch.mean(XX + YY - XY -YX)
    return loss

source_device_id = source_device
target_device_id = target_device

if(source_device==1):
    real_key_01 = 0x21 # key of the source domain
elif(source_device==2):
    real_key_01 = 0xCD # key of the source domain
elif(source_device==3):
    real_key_01 = 0x8F # key of the source domain

if(target_device==1):
    real_key_02 = 0x01 # key of the target domain
elif(target_device==2):
    real_key_02 = 0x02 # key of the target domain
elif(target_device==3):
    real_key_02 = 0x03 # key of the target domain
elif(target_device==4):
    real_key_02 = 0x04 # key of the target domain
elif(target_device==5):
    real_key_02 = 0x05 # key of the target domain
elif(target_device==6):
    real_key_02 = 0x06 # key of the target domain
elif(target_device==7):
    real_key_02 = 0x07 # key of the target domain
elif(target_device==8):
    real_key_02 = 0x08 # key of the target domain

lambda_ = 0.1 # Penalty coefficient
labeling_method = 'identity' # labeling of trace
preprocess = 'horizontal_standardization' # preprocess method
batch_size = 50
total_epoch = 100
finetune_epoch = 50 # epoch number for fine-tuning
lambda1 = 2
lambda2 = 0.05
lambda3 = 0 # penalty coefficients for MMD loss (stability
log_interval = 20 # epoch interval to log training information
train_num = 20000
source_test_num = 5000
trace_offset = 0
# params for optimizing models
d_learning_rate = 1e-3
c_learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.9
trace_num_max = 1000

target_test_num = 5000
trace_offset = 0
target_trace_length = 1500
source_trace_length = 1000
source_file_path = './Data/device'+str(source_device)+'/'
target_file_path = '../XMEGA-EM/Data/device0'+str(target_device)+'/'

no_cuda =False
cuda = not no_cuda and torch.cuda.is_available()
seed = 42
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

class_num = 256
labeling_method = 'identity' # labeling of trace
train_num_fix = 20000
target_finetune_num = 20000

# to load traces and labels
X_train_source = np.load(source_file_path + 'X_train.npy')
Y_train_source = np.load(source_file_path + 'Y_train.npy')
X_attack_source = np.load(source_file_path + 'X_attack.npy')
Y_attack_source = np.load(source_file_path + 'Y_attack.npy')
X_train_target = np.load(target_file_path + 'X_train.npy')
Y_train_target = np.load(target_file_path + 'Y_ID_train.npy')
X_attack_target = np.load(target_file_path + 'X_attack.npy')
Y_attack_target = np.load(target_file_path + 'Y_ID_attack.npy')

# to load ciphertexts
plaintexts_target_train = np.load(target_file_path + 'plaintexts_train.npy')
plaintexts_target_attack = np.load(target_file_path + 'plaintexts_attack.npy')

#breakpoint()

mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
X_train_source = (X_train_source - mn)/std

mn = np.repeat(np.mean(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
std = np.repeat(np.std(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
X_attack_source = (X_attack_source - mn)/std

mn = np.repeat(np.mean(X_train_target, axis=1, keepdims=True), X_train_target.shape[1], axis=1)
std = np.repeat(np.std(X_train_target, axis=1, keepdims=True), X_train_target.shape[1], axis=1)
X_train_target = (X_train_target - mn)/std

mn = np.repeat(np.mean(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
std = np.repeat(np.std(X_attack_target, axis=1, keepdims=True), X_attack_target.shape[1], axis=1)
X_attack_target = (X_attack_target - mn)/std

# parameters of data loader
# parameters of data loader
kwargs_source_train = {
        'trs_file': X_train_source[0:train_num,:],
        'label_file': Y_train_source[0:train_num],
        'trace_num':train_num,
        'trace_offset':trace_offset,
        'trace_length':source_trace_length,
}
kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num':source_test_num,
        'trace_offset':trace_offset,
        'trace_length':source_trace_length,
}
kwargs_target_finetune = {
        'trs_file': X_train_target[0:target_finetune_num,:],
        'label_file': Y_train_target[0:target_finetune_num],
        'trace_num':target_finetune_num,
        'trace_offset':trace_offset,
        'trace_length':target_trace_length,
}
kwargs_target = {
        'trs_file': X_attack_target,
        'label_file': Y_attack_target,
        'trace_num':target_test_num,
        'trace_offset':trace_offset,
        'trace_length':target_trace_length,
}
source_train_loader = load_training(batch_size, kwargs_source_train)
source_test_loader = load_testing(batch_size, kwargs_source_test)
target_finetune_loader = load_training(batch_size, kwargs_target_finetune)
target_test_loader = load_testing(batch_size, kwargs_target)
print('Load data complete!')

### the pre-trained model
class UTLA_Net(nn.Module):
    def __init__(self, num_classes=class_num):
        super(UTLA_Net, self).__init__()
        # the encoder part
        self.features_1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=2, stride=2))
        self.features_2 = nn.Sequential(    
            nn.Conv1d(8, 16, kernel_size=9),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=9, stride=9))
        self.features_3 = nn.Sequential(  
            nn.Conv1d(16, 32, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=3, stride=3),
        )
        self.features_4 = nn.Sequential(  
            nn.Conv1d(32, 64, kernel_size=2),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        # the fully-connected layer 1
        self.classifier_1 = nn.Sequential(
            nn.Linear(192, 2),
            nn.SELU(),
        )
        # the output layer
        self.final_classifier = nn.Sequential(
            nn.Linear(2, num_classes)
        )
    # how the network runs
    def forward(self, target):
        #target data flow
        if(target.shape[2] == 1000): 
            target = target[:, :, -500:]
        elif(target.shape[2] == 1500):
            target = target[:, :, 450:450+500]
        target = self.features_1(target)
        target_0 = self.features_2(target)
        target_1 = self.features_3(target_0)
        target_2 = self.features_4(target_1)
        target_2 = target_2.view(target_2.size(0), -1)
        target_3 = self.classifier_1(target_2)
        result = self.final_classifier(target_3)
        return result, target_2, target_1.view(target_1.size(0), -1),target_0.view(target_0.size(0), -1)

### the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # the discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(192, 64),
            nn.SELU(),
            nn.Linear(64, 2)
        )
    # how the network runs
    def forward(self, input):
        output = self.discriminator(input)
        return output


# Randomly re-initialize features_1, features_2, features_3
def weights_init_random(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
        nn.init.constant_(m.bias, 1)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 1)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
        nn.init.constant_(m.bias, 1)

# create a target network
UTLA_training_model = UTLA_Net(num_classes=class_num)
UTLA_profile_model = UTLA_Net(num_classes=class_num)
discriminator = Discriminator()
if cuda:
    UTLA_training_model.cuda()
    UTLA_profile_model.cuda()
    discriminator.cuda()

# initialize a big enough loss
min_loss = 1000
Intermediate_GE = np.zeros([finetune_epoch])
Intermediate_NTGE = np.zeros([finetune_epoch])

optimizer_critic = optim.SGD([
        {'params': discriminator.discriminator.parameters()},
    ],lr=d_learning_rate, weight_decay=0.0005, momentum=0.9)
optimizer_model = optim.SGD([
        {'params': UTLA_training_model.features_1.parameters()},
        {'params': UTLA_training_model.features_2.parameters()},
        {'params': UTLA_training_model.features_3.parameters()},
        {'params': UTLA_training_model.features_4.parameters()},
    ],lr=c_learning_rate, weight_decay=0.0005, momentum=0.9)

# load the pre-trained network
if(set_UTLA_train==1):
    print("Start-training")
    #initialization-profile
    checkpoint_profile = torch.load('./models/pre-trained_device{}.pth'.format(source_device_id),weights_only=True)
    model_dict = checkpoint_profile['model_state_dict']
    UTLA_profile_model.load_state_dict(model_dict)

    #initialization-target
    #checkpoint_target = torch.load('./models/pre-trained_device_{}_{}.pth'.format(source_device_id, target_device_id),weights_only=True)
    checkpoint_target = torch.load('./models/pre-trained_device{}.pth'.format(source_device_id),weights_only=True)
    model_dict = checkpoint_target['model_state_dict']
    #UTLA_training_model.load_state_dict(model_dict)
    # Filter only classifier_1 and final_classifier layers
    filtered_state_dict = {k: v for k, v in model_dict.items() if 'classifier_1' in k or 'final_classifier' in k}
    #filtered_state_dict = {k: v for k, v in model_dict.items() if 'upsample' not in k}
    # Load only matching parameters
    UTLA_training_model.load_state_dict(filtered_state_dict, strict=False)
    UTLA_training_model.features_1.apply(weights_init_random)
    UTLA_training_model.features_2.apply(weights_init_random)
    UTLA_training_model.features_3.apply(weights_init_random)
    UTLA_training_model.features_4.apply(weights_init_random)

    # restore the optimizer state
    for epoch in range(1, finetune_epoch + 1):
        print(f'Train Epoch {epoch}:')
        UTLA_train(epoch, UTLA_training_model, UTLA_profile_model, discriminator)
        Intermediate_NTGE[epoch-1], Intermediate_GE[epoch-1] = test_intermediate(UTLA_training_model)
        #encoder_rep(epoch, UTLA_training_model, UTLA_profile_model)
        print('Intermediate NTGE at epoch %d: %d' % (epoch, Intermediate_NTGE[epoch-1])) # print the intermediate NTGE for debugging
        print('Intermediate GE at epoch %d: %.2f' % (epoch, Intermediate_GE[epoch-1])) # print the intermediate GE for debugging
        #Plotting Intermediate GE Val
        plt.figure(figsize=(6,4))
        plt.plot(Intermediate_GE,color='red')
        ax = plt.gca()
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel('GE(%d)' % (trace_num_max), fontsize = 15) # GE for the max trace number') 
        plt.ylim((0, 256))
        ax.tick_params(axis='y', labelsize=12) 
        ax.tick_params(axis='x', labelsize=12)      
        plt.tight_layout()  # Automatically adjust padding to prevent clipping  
        plt.show()
        plt.savefig('./figures/GE_UTLA_crossSC_XMEGA_{}_to_{}_'.format(source_device_id, target_device_id) + '_training' + '.png') 
        np.save('./results/GE_UTLA_crossSC_XMEGA_{}_to_{}_'.format(source_device_id, target_device_id) + '_training', Intermediate_GE)

    torch.save({
        'epoch': epoch,
        'model_state_dict': UTLA_training_model.state_dict(),
        }, './models/UTLA-final_crossSC_XMEGA_device{}_to_{}.pth'.format(source_device_id, target_device_id))
    
    plt.figure(figsize=(6,4))
    plt.plot(Intermediate_NTGE,color='red')
    ax = plt.gca()
    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel(r'$N_{TGE}$', fontsize = 15)
    ax.tick_params(axis='y', labelsize=12) 
    ax.tick_params(axis='x', labelsize=12)   
    plt.tight_layout()  # Automatically adjust padding to prevent clipping   
    plt.show()
    plt.savefig('./figures/NTGE_UTLA_crossSC_XMEGA_{}_{}_'.format(source_device_id, target_device_id) + '_training' + '.png') 
    np.save('./results/NTGE_UTLA_crossSC_XMEGA_{}_{}_'.format(source_device_id, target_device_id) + '_training', Intermediate_NTGE)    
    print('Final NTGE',Intermediate_NTGE[-1]) # print the minimum Intermediate GE for debugging
    print('Final GE',Intermediate_GE[-1]) # print the minimum Intermediate GE for debugging

# create a network
UTLA_test_model = UTLA_Net(num_classes=class_num)
print('Construct model complete')
if cuda:
    UTLA_test_model.cuda()

checkpoint = torch.load('./models/UTLA-final_crossSC_XMEGA_device{}_to_{}.pth'.format(source_device_id, target_device_id),weights_only=True)

model_dict = checkpoint['model_state_dict']
UTLA_test_model.load_state_dict(model_dict)

# evaluate the final model on source and target domain
with torch.no_grad():
    print('Result on target device:')
    test(UTLA_test_model, UTLA_profile_model, target_device_id, model_flag='UTLA_target')
