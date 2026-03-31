labeling_method = "identity"
preprocess = "horizontal_standardization"

batch_size = 50
total_epoch = 100
finetune_epoch = 50
log_interval = 20

train_num = 20000
valid_num = 5000
source_test_num = 9900
target_finetune_num = 20000
target_test_num = 5000

trace_offset = 0
trace_length = 1000
trace_num_max = 1000

lambda1 = 2.0
lambda2 = 0.05
lambda3 = 0.0

d_learning_rate = 1e-3
c_learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.9
