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
import h5py

# main
train_first_time= int(sys.argv[1])
print_intermediate_GE = int(sys.argv[2]) 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def env_int(name, default):
    return int(os.environ.get(name, default))

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

### To train a network 
def train(epoch, model):
    """
    - epoch : the current epoch
    - model : the current model   
    - freeze_BN : whether to freeze batch normalization layers
    """
    model.train() # enter training mode 
    # Instantiate the Iterator
    iter_source = iter(source_train_loader)
    # get the number of batches
    num_iter = len(source_train_loader)
    clf_criterion = nn.CrossEntropyLoss()
    # train on each batch of data
    for i in range(1, num_iter+1):
        source_data, source_label, source_idx = next(iter_source)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        plaintext_batch = plaintexts_source_attack[source_idx]
        real_key_batch = real_key_profile[source_idx]
        optimizer.zero_grad()
        source_preds = model(source_data)
        softmax = nn.Softmax(dim=1)
        preds = source_preds.data.max(1, keepdim=True)[1]
        correct_batch = preds.eq(source_label.data.view_as(preds)).sum()
        loss = GE_diff_loss(source_preds, plaintext_batch, real_key_batch)
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            #breakpoint()
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}%'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                100. * i / len(source_train_loader), loss.data, float(correct_batch) * 100. /batch_size))

def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def GE_diff_loss(source_preds, plaintext_batch, real_key_batch, trace_num_max=None):
    """
    Fully differentiable Guessing Entropy (GE) loss approximation.
    
    Args:
        source_preds: [B, 2] tensor - model output after softmax & compute_Pr2
        plaintext_batch: [B] uint8 tensor
        Sbox: list or 1D tensor of 256 inverse S-box values
        real_key: int
        trace_num_max: number of traces to simulate (default 2000)
    Returns:
        GE_loss: differentiable scalar
    """
    device = source_preds.device
    if trace_num_max is None:
        trace_num_max = env_int("UTLA_GE_LOSS_TRACE_NUM_MAX", 2000)
    # Initialize the prediction and label lists(tensors)
    if not isinstance(plaintext_batch, torch.Tensor):
        plaintext_batch = torch.tensor(plaintext_batch, dtype=torch.uint8)
    
    # Ensure ciphertext is on the same device and correct type
    plaintext_batch = plaintext_batch.to(source_preds.device).to(torch.uint8)

    if not isinstance(real_key_batch, torch.Tensor):
        real_key_batch = torch.tensor(real_key_batch, dtype=torch.uint8)
    
    # Ensure ciphertext is on the same device and correct type
    real_key_batch = real_key_batch.to(source_preds.device).to(torch.uint8)

    B = plaintext_batch.shape[0]
    K = 256  # key guesses

    # Convert InvSbox to tensor if needed
    sbox = torch.tensor(Sbox, dtype=torch.long, device=device)

    keys = torch.arange(0, K, device=device, dtype=torch.uint8)  # [256]

    # Compute guessed labels [B, 256]
    guessed_states = plaintext_batch.unsqueeze(1) ^ keys.view(1, -1)  # [B, 256]
    guessed_states = guessed_states.long()
    guessed_labels = sbox[guessed_states]  # [B, 256]

    # Compute true labels [B]
    true_states = plaintext_batch ^ real_key_batch
    true_states = true_states.long()
    true_labels = sbox[true_states]
    
    # Get model confidence scores for guessed vs true labels
    guessed_probs = torch.gather(source_preds.unsqueeze(1).expand(-1, K, -1), 2, guessed_labels.unsqueeze(2)).squeeze(2)  # [B, 256]
    true_probs = source_preds[torch.arange(B), true_labels]  # [B]
    score_mat = guessed_probs - true_probs.unsqueeze(1)  # [B, 256]
    score_mat_neg = -score_mat

    # Compute mean and variance for each key guess
    mean_est = score_mat.mean(dim=0)  # [256]
    var_est = score_mat.var(dim=0, unbiased=False) + 1e-6  # [256]
    mean_est_neg = score_mat_neg.mean(dim=0)
    var_est_neg = score_mat_neg.var(dim=0, unbiased=False) + 1e-6

    # Simulate GE evolution over traces
    trace_range = torch.arange(1, trace_num_max + 1, device=device).float().unsqueeze(1)  # [T, 1]
    scale = torch.sqrt(trace_range)  # [T, 1]

    scaled_pos = scale * mean_est.view(1, -1) / var_est.view(1, -1).sqrt()  # [T, 256]
    scaled_neg = scale * mean_est_neg.view(1, -1) / var_est_neg.view(1, -1).sqrt()

    GE_pos = torch.sum(normal_cdf(scaled_pos), dim=1)  # [T]
    GE_neg = torch.sum(normal_cdf(scaled_neg), dim=1)

    # Pick the smaller GE at the final trace count
    GE_final = torch.min(GE_pos[-1], GE_neg[-1])

    return GE_final  # Differentiable scalar

def test_intermediate(model):
    # enter evaluation mode
    model.eval()
    real_key = real_key_profile
    # Initialize the prediction and label lists(tensors)
    iter_source = iter(source_train_loader)

    # max trace num for attack
    trace_num_max = env_int("UTLA_TRACE_NUM_MAX", 1000)
    num_iter = int(trace_num_max/batch_size)
    guessing_entropy = np.zeros(trace_num_max)
    mean_est = np.zeros([256])
    var_est = np.zeros([256])
    guessing_entropy_neg = np.zeros(trace_num_max)
    mean_est_neg = np.zeros([256])
    var_est_neg = np.zeros([256])
    score_mat = np.zeros((trace_num_max, 256))
    plaintext = plaintexts_source_train
    test_preds_all = torch.zeros((trace_num_max, class_num), dtype=torch.float, device='cpu')
    ordering = np.zeros(trace_num_max)

    for i in range(0,num_iter):
        source_data, source_label, source_idx = next(iter_source)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        source_preds = model(source_data)
        softmax = nn.Softmax(dim=1)
        test_preds_all[i*batch_size:(i+1)*batch_size, :] = softmax(source_preds)
        ordering[i*batch_size:(i+1)*batch_size] = source_idx

    #breakpoint()
    for i in range(0,trace_num_max):
        trueState = plaintext[int(ordering[i])] ^ real_key[int(ordering[i])]
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
            if(fk!=real_key[i]):
                guessing_entropy[i] = guessing_entropy[i]  + norm.cdf(np.sqrt(i+1)*mean_est[fk]/np.sqrt(var_est[fk]), loc=0, scale=1)
                guessing_entropy_neg[i] = guessing_entropy_neg[i]  + norm.cdf(np.sqrt(i+1)*mean_est_neg[fk]/np.sqrt(var_est_neg[fk]), loc=0, scale=1)

    if(guessing_entropy_neg[-1] < guessing_entropy[-1]):
        # if negative score is better, use negative score
        guessing_entropy = guessing_entropy_neg
    
    guessing_entropy = guessing_entropy.astype(int)
    if(np.size(np.where(guessing_entropy<2))==0):
        output_str = np.argmin(guessing_entropy) 
    else:
        # find the first point where GE < 2
        output_str = np.where(guessing_entropy<2)[0][0]
    #breakpoint()
    return output_str+1, guessing_entropy[-1]
           
### test/attack
def test(model, device_id, disp_GE=True, model_flag='pretrained'):
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
    epoch = 0
    clf_criterion = nn.CrossEntropyLoss()
    test_num = source_test_num
    test_loader = source_test_loader
    real_key = real_key_attack
    test_preds_all = torch.zeros((test_num, class_num), dtype=torch.float, device='cpu')
    for data, label, idx in test_loader:
        if cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        test_preds = model(data)
        # sum up batch loss
        test_loss += clf_criterion(test_preds, label) 
        # get the index of the max probability
        pred = test_preds.data.max(1)[1]
        softmax = nn.Softmax(dim=1)
        # get the softmax results for attack/showing guessing entropy
        test_preds_all[epoch*batch_size:(epoch+1)*batch_size, :] = softmax(test_preds)
        # get the number of correct prediction
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        epoch += 1
    test_loss /= len(test_loader)
    print('Target test loss: {:.4f}, Target test accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss.data, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # show the guessing entropy and success rate
    if disp_GE:
        plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)

### show the guessing entropy 
def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    """
    - preds : the probability for each class (n*256 for a byte)
    - real_key : the key of the target device
    - device_id : id of the target device
    - model_flag : a string for naming GE result
    """
    # max trace num for attack
    trace_num_max = env_int("UTLA_TRACE_NUM_MAX", 1000)
    guessing_entropy = np.zeros(trace_num_max)
    guessing_entropy_neg = np.zeros(trace_num_max)
    mean_est = np.zeros([256])
    var_est = np.zeros([256])
    mean_est_neg = np.zeros([256])
    var_est_neg = np.zeros([256])
    score_mat = np.zeros((trace_num_max, 256))
    plaintext = plaintexts_source_attack

    for i in range(0,trace_num_max):
        for key_guess in range(0, 256):
            initialState = plaintext[i] ^ key_guess
            label = Sbox[initialState]
            trueState = plaintext[i] ^ real_key[i]
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
            if(fk!=real_key[i]):
                guessing_entropy[i] = guessing_entropy[i]  + norm.cdf(np.sqrt(i+1)*mean_est[fk]/np.sqrt(var_est[fk]), loc=0, scale=1)
                guessing_entropy_neg[i] = guessing_entropy_neg[i]  + norm.cdf(np.sqrt(i+1)*mean_est_neg[fk]/np.sqrt(var_est_neg[fk]), loc=0, scale=1)

    if(guessing_entropy_neg[-1] < guessing_entropy[-1]):
        # if negative score is better, use negative score
        guessing_entropy = guessing_entropy_neg
    
    guessing_entropy = guessing_entropy.astype(int)
    if(np.size(np.where(guessing_entropy<2))==0):
        output_str = np.argmin(guessing_entropy)
    else:
        # find the first point where GE < 2
        output_str = np.where(guessing_entropy<2)[0][0]
    #breakpoint()
    print(output_str+1, guessing_entropy[-1])
    plt.figure(figsize=(6,4))
    p1, = plt.plot(guessing_entropy,color='red')
    plt.xlabel('Number of traces')
    plt.ylabel('Guessing entropy')
    plt.ylim((0, 128))
    plt.show()
    plt.savefig('./figures/entropy_ASCADv1_wGE_{}_to_{}_'.format(source_device_id, device_id) + model_flag + '.png') 
    np.save('./results/entropy_ASCADv1_wGE_{}_to_{}_'.format(source_device_id, device_id) + model_flag, guessing_entropy)

source_device_id = 1
target_device_id = 1

target_file	 = h5py.File('ascad-variable.h5', "r")

lambda_ = 0.1 # Penalty coefficient
labeling_method = 'identity' # labeling of trace
preprocess = 'horizontal_standardization' # preprocess method
batch_size = 50
batch_size = env_int("UTLA_BATCH_SIZE", batch_size)
total_epoch = env_int("UTLA_TOTAL_EPOCHS", 100)
finetune_epoch = env_int("UTLA_FINETUNE_EPOCHS", 50) # epoch number for fine-tuning
log_interval = env_int("UTLA_LOG_INTERVAL", 20) # epoch interval to log training information
train_num = env_int("UTLA_TRAIN_NUM", 20000)
source_test_num = env_int("UTLA_SOURCE_TEST_NUM", 10000)
trace_offset = 0
lr = 0.0001
no_cuda =False
cuda = not no_cuda and torch.cuda.is_available()
seed = 42
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
class_num = 256
trace_num_max = env_int("UTLA_TRACE_NUM_MAX", 1000)
source_test_num = max(source_test_num, trace_num_max)
source_trace_length = 1400

no_cuda =False
cuda = not no_cuda and torch.cuda.is_available()
seed = 42
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
class_num = 256
trace_num_max = env_int("UTLA_TRACE_NUM_MAX", trace_num_max)

X_train_source = np.array(target_file['Profiling_traces/traces'][:train_num], dtype=np.int8)
X_attack_source = np.array(target_file['Attack_traces/traces'][:source_test_num], dtype=np.int8)
Y_train_source = np.array(target_file['Profiling_traces/labels'][:train_num])
Y_attack_source = np.array(target_file['Attack_traces/labels'][:source_test_num])

# to load ciphertexts
real_key_profile = target_file['Profiling_traces/metadata']['key'][:train_num,2]
real_key_attack = target_file['Attack_traces/metadata']['key'][:source_test_num,2]
plaintexts_source_attack = target_file['Attack_traces/metadata']['plaintext'][:source_test_num,2]
plaintexts_source_train = target_file['Profiling_traces/metadata']['plaintext'][:train_num,2]

mn = np.repeat(np.mean(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
std = np.repeat(np.std(X_train_source, axis=1, keepdims=True), X_train_source.shape[1], axis=1)
X_train_source = (X_train_source - mn)/std

mn = np.repeat(np.mean(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
std = np.repeat(np.std(X_attack_source, axis=1, keepdims=True), X_attack_source.shape[1], axis=1)
X_attack_source = (X_attack_source - mn)/std

# parameters of data loader
kwargs_source_train = {
        'trs_file': X_train_source[0:train_num,:],
        'label_file': Y_train_source[0:train_num],
        'trace_num':train_num,
        'trace_offset':trace_offset,
        'trace_length':source_trace_length,
}
source_train_loader = load_training(batch_size, kwargs_source_train)

kwargs_source_test = {
        'trs_file': X_attack_source,
        'label_file': Y_attack_source,
        'trace_num':source_test_num,
        'trace_offset':trace_offset,
        'trace_length':source_trace_length,
}

source_train_loader = load_training(batch_size, kwargs_source_train)
source_test_loader = load_testing(batch_size, kwargs_source_test)

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
            nn.Linear(768, 20),
            nn.SELU(),
        )
        # the fully-connected layer 2
        self.classifier_2 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
        )
        # the fully-connected layer 3
        self.classifier_3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU()
        )
        # the output layer
        self.final_classifier = nn.Sequential(
            nn.Linear(20, num_classes)
        )
     # how the network runs
    def forward(self, input):
        #target data flow
        #input = input[:, :, -500:]
        x = self.features_1(input)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = x.view(x.size(0), -1)
        target = self.classifier_1(x)
        target = self.classifier_2(target)
        target = self.classifier_3(target)
        result = self.final_classifier(target)
        return result
    
# create a network
Profile_model = UTLA_Net(num_classes=class_num)
print('Construct model complete')
if cuda:
    Profile_model.cuda()

Intermediate_GE = np.zeros(total_epoch)
Intermediate_NTGE = np.zeros(total_epoch)
if(train_first_time==1):
    # initialize a big enough loss
    min_loss = 1000
    optimizer = optim.Adam([
        {'params': Profile_model.parameters()},
    ], lr=lr)
    # restore the optimizer state

    for epoch in range(1, total_epoch + 1):
        print(f'Train Epoch {epoch}:')
        train(epoch, Profile_model)
        if(print_intermediate_GE == 1):

            Intermediate_NTGE[epoch-1], Intermediate_GE[epoch-1]  = test_intermediate(Profile_model) # get the intermediate GE and NTGE for the current epoch
            print('Intermediate GE(%d): %d' % (trace_num_max, Intermediate_GE[epoch-1]))
            print('Intermediate NTGE(%d): %d' % (trace_num_max, Intermediate_NTGE[epoch-1]))

            plt.figure(figsize=(6,4))
            plt.plot(Intermediate_GE,color='red')
            ax = plt.gca()
            plt.xlabel('Epoch', fontsize = 15)
            plt.ylabel('GE(%d)' % (trace_num_max), fontsize = 15) # GE for the max trace number') 
            #plt.ylim((0, 128))
            ax.tick_params(axis='y', labelsize=12) 
            ax.tick_params(axis='x', labelsize=12)  
            plt.tight_layout()  # Automatically adjust padding to prevent clipping    
            plt.show()
            plt.savefig('./figures/GE_ASCADv1_wGE_{}_'.format(source_device_id) + '_training' + '.png') 
            np.save('./results/GE_ASCADv1_wGE_{}_'.format(source_device_id) + '_training', Intermediate_GE)
            
    torch.save({
        'epoch': epoch,
        'model_state_dict': Profile_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, './models/pre-trained_device_ASCADv1_wGE{}.pth'.format(source_device_id))

    if(print_intermediate_GE == 1):    
        

        plt.figure(figsize=(6,4))
        plt.plot(Intermediate_NTGE,color='red')
        ax = plt.gca()
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel(r'$N_{TGE}$', fontsize = 15)
        #plt.ylim((0, 500))
        ax.tick_params(axis='y', labelsize=12) 
        ax.tick_params(axis='x', labelsize=12)   
        plt.tight_layout()  # Automatically adjust padding to prevent clipping 
        plt.show()
        plt.savefig('./figures/NTGE_ASCADv1_wGE_{}_'.format(source_device_id) + '_training' + '.png') 
        np.save('./results/NTGE_ASCADv1_wGE_{}_'.format(source_device_id) + '_training', Intermediate_NTGE)
        
        print('Final NTGE',Intermediate_NTGE[-1]) # print the minimum Intermediate GE for debugging
        print('Final GE',Intermediate_GE[-1]) # print the minimum Intermediate GE for debugging

else:        
    # load the pre-trained network
    checkpoint = torch.load('./models/pre-trained_device_ASCADv1_wGE{}.pth'.format(source_device_id),weights_only=True)
    pretrained_dict = checkpoint['model_state_dict']
    Profile_model.load_state_dict(pretrained_dict)


# evaluate the pre-trained model on source and target domain
with torch.no_grad():
    print('Result on target device:')
    test(Profile_model, target_device_id, model_flag='pretrained_target')
