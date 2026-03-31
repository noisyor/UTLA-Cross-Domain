"""Params for ADDA."""

lambda_ = 0.1 # Penalty coefficient
labeling_method = 'identity' # labeling of trace
preprocess = 'horizontal_standardization' # preprocess method
batch_size = 100
total_epoch = 100
finetune_epoch = 10 # epoch number for fine-tuning
_lambda1 = 0.05
_lambda2 = 2
log_interval = 40 # epoch interval to log training information
train_num = 20000
valid_num = 5000
source_test_num = 5000
target_finetune_num = 100
target_test_num = 4500
trace_offset = 0
trace_length = 500

# params for optimizing models
d_learning_rate = 1e-5
c_learning_rate = 1e-5
beta1 = 0.5
beta2 = 0.9
