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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    return train_loader

### data loader for testing
def load_testing(batch_size, kwargs):
    data = TorchDataset(**kwargs)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    return test_loader

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

def compute_Pr2(source_pred):
    Pr2 = torch.zeros((source_pred.size(0), 2), device=source_pred.device)

    # Calculate Pr2[:, 0] (for i = 1)
    Pr2[:, 0] = torch.sum(source_pred[:, 0::16], dim=1)

    # Pr2[:, 1] is 1 - Pr2[:, 0]
    Pr2[:, 1] = 1 - Pr2[:, 0]

    return Pr2

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
    #breakpoint()
    # train on each batch of data
    for i in range(1, num_iter+1):
        source_data, source_label, source_idx = next(iter_source)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        cipher_batch = ciphertexts_source_train[source_idx]
        optimizer.zero_grad()
        source_preds = model(source_data)
        softmax = nn.Softmax(dim=1) # softmax layer to compute probabilities
        source_preds = softmax(source_preds)
        source_preds = compute_Pr2(source_preds) # compute Pr2 for the output of the model
        preds = source_preds.data.max(1, keepdim=True)[1]
        correct_batch = preds.eq(source_label.data.view_as(preds)).sum()
        loss = GE_diff_loss(source_preds, cipher_batch)
        loss.backward()
        optimizer.step()
        #breakpoint()
        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}%'.format(
                epoch, i * len(source_data), len(source_train_loader) * batch_size,
                100. * i / len(source_train_loader), loss.data, float(correct_batch) * 100. /batch_size))


def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def GE_diff_loss(source_preds, cipher_batch, trace_num_max=2000):
    """
    Fully differentiable Guessing Entropy (GE) loss approximation.
    
    Args:
        source_preds: [B, 2] tensor - model output after softmax & compute_Pr2
        cipher_batch: [B, 16] uint8 tensor
        InvSbox: list or 1D tensor of 256 inverse S-box values
        real_key: int
        trace_num_max: number of traces to simulate (default 2000)
    Returns:
        GE_loss: differentiable scalar
    """
    device = source_preds.device
    # Initialize the prediction and label lists(tensors)
    if not isinstance(cipher_batch, torch.Tensor):
        cipher_batch = torch.tensor(cipher_batch, dtype=torch.uint8)

    # Ensure ciphertext is on the same device and correct type
    cipher_batch = cipher_batch.to(source_preds.device).to(torch.uint8)

    B = cipher_batch.shape[0]
    K = 256  # key guesses

    # Convert InvSbox to tensor if needed
    inv_sbox = torch.tensor(InvSbox, dtype=torch.uint8, device=device)

    ct14 = cipher_batch[:, 14]  # [B]
    keys = torch.arange(0, K, device=device, dtype=torch.uint8)  # [256]

    # Compute guessed labels [B, 256]
    guessed_states = ct14.unsqueeze(1) ^ keys.view(1, -1)  # [B, 256]
    guessed_states = guessed_states.long()
    guessed_inv = inv_sbox[guessed_states]  # [B, 256]
    guessed_labels = ((guessed_inv // 16) == 0).long()  # [B, 256]

    # Compute true labels [B]
    true_states = ct14 ^ real_key_01
    true_states = true_states.long()
    true_inv = inv_sbox[true_states]
    true_labels = ((true_inv // 16) == 0).long()  # [B]

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

           
### test/attack
def test(model, device_id, model_flag='pretrained'):
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
    real_key = real_key_01
    test_preds_all = torch.zeros((test_num, 2), dtype=torch.float, device='cpu')
    for data, label, idx in test_loader:
        if cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        test_preds = model(data)
        softmax = nn.Softmax(dim=1) # softmax layer to compute probabilities
        test_preds = softmax(test_preds)
        test_preds = compute_Pr2(test_preds) # compute Pr2 for the output of the model
        test_loss += clf_criterion(test_preds, label) 
        # get the index of the max probability
        pred = test_preds.data.max(1)[1]
        # get the softmax results for attack/showing guessing entropy
        test_preds_all[epoch*batch_size:(epoch+1)*batch_size, :] = test_preds
        # get the number of correct prediction
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        epoch += 1
    test_loss /= len(test_loader)
    print('source test loss: {:.4f}, source test accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss.data, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # show the guessing entropy and success rate
    plot_guessing_entropy(test_preds_all.numpy(), real_key, device_id, model_flag)

### show the guessing entropy 
def plot_guessing_entropy(preds, real_key, device_id, model_flag):
    """
    - preds : the probability for each class (n*256 for a byte)
    - real_key : the key of the source device
    - device_id : id of the source device
    - model_flag : a string for naming GE result
    """
    # max trace num for attack
    guessing_entropy = np.zeros(trace_num_max)
    guessing_entropy_neg = np.zeros(trace_num_max)
    mean_est = np.zeros([256])
    var_est = np.zeros([256])
    mean_est_neg = np.zeros([256])
    var_est_neg = np.zeros([256])
    score_mat = np.zeros((trace_num_max, 256))
    ciphertext = ciphertexts_source_attack
   
    for i in range(0,trace_num_max):
        for key_guess in range(0, 256):
            initialState = int(ciphertext[i,14]) ^ key_guess
            label = int(InvSbox[initialState]  // 16 == 0)
            trueState = int(ciphertext[i,14]) ^ real_key
            truelabel = int(InvSbox[trueState] // 16 == 0)
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

    guessing_entropy = guessing_entropy.astype(int)
    if(np.size(np.where(guessing_entropy<2))==0):
        output_str = trace_num_max -1 
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
    plt.savefig('./figures/GE_'+ labeling_method + '_{}_to_{}_'.format(source_device_id, device_id) + model_flag + '.png') 
    np.save('./results/GE_'+ labeling_method + '_{}_to_{}_'.format(source_device_id, device_id) + model_flag, guessing_entropy)

source_device_id = 1

real_key_01 = 48 # key of the source domain

in_file	 = h5py.File('dataset.h5', "r")

attack_set = in_file['attack_dataset']

train_set = in_file['training_dataset']

labeling_method = 'identity' # labeling of trace
batch_size = 100
total_epoch = 100
finetune_epoch = 15 # epoch number for fine-tuning
lr = 0.001 # learning rate
log_interval = 50 # epoch interval to log training information
train_num = 75000
valid_num = 5000
source_test_num = 20000
trace_offset = 0
source_trace_length = 250
trace_num_max = 2000
no_cuda =False
cuda = not no_cuda and torch.cuda.is_available()
seed = 42
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
class_num = 256

# to load traces and labels
X_train_source = train_set['traces']
Y_train_source = train_set['labels'][:,real_key_01]
X_attack_source = attack_set['traces']
Y_attack_source = attack_set['labels'][:,real_key_01]

# to load ciphertexts
ciphertexts_source_attack = attack_set['ciphertext']
ciphertexts_source_train = train_set['ciphertext']
#breakpoint()

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

source_test_loader = load_testing(batch_size, kwargs_source_test)

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
            nn.Linear(64, 20),
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
    def forward(self, source):

        #source data flow
        if(source.shape[2] == 1500): # for compatibility with the source domain    
            x = source[:, :, -1400:]
        else:
            x = source
        x = self.features_1(x)
        source_0 = self.features_2(x)
        source_1 = self.features_3(source_0)
        source_2 = self.features_4(source_1)
        source_2 = source_2.view(source_2.size(0), -1)
        source_3 = self.classifier_1(source_2)
        source_3 = self.classifier_2(source_3)
        source_3 = self.classifier_3(source_3)
        result = self.final_classifier(source_3)
        return result

# create a network
Profile_model = UTLA_Net(num_classes=class_num)
print('Construct model complete')
if cuda:
    Profile_model.cuda()

optimizer = optim.Adam([{'params': Profile_model.parameters()},], lr=lr)
if(train_first_time==1):
    # initialize a big enough loss
    min_loss = 1000

    for epoch in range(1, total_epoch + 1):
        print(f'Train Epoch {epoch}:')
        train(epoch, Profile_model)
            
    torch.save({
        'epoch': epoch,
        'model_state_dict': Profile_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, './models/pre-trained_device{}.pth'.format(source_device_id)) 
        
else:        
    # load the pre-trained network
    checkpoint = torch.load('./models/pre-trained_device{}.pth'.format(source_device_id),weights_only=True)
    pretrained_dict = checkpoint['model_state_dict']
    Profile_model.load_state_dict(pretrained_dict)

# evaluate the pre-trained model on source and source domain
with torch.no_grad():
    print('Result on source device:')
    test(Profile_model, source_device_id, model_flag='pretrained_source')