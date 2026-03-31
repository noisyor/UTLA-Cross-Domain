"""Params for ADDA."""

lambda_ = 0.1 # Penalty coefficient
labeling_method = 'identity' # labeling of trace
preprocess = 'horizontal_standardization' # preprocess method
batch_size = 50
total_epoch = 100
finetune_epoch = 20 # epoch number for fine-tuning
_lambda1 = 0.05
_lambda2 = 2
lambda1 = 2.0
lambda2 = 0.05
log_interval = 50 # epoch interval to log training information
train_num = 45000
valid_num = 5000
source_test_num = 10000
target_finetune_num = 45000
target_test_num = 10000
trace_offset = 0
trace_length = 700

# params for optimizing models
d_learning_rate = 1e-5
c_learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.9
