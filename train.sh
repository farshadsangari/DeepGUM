# Parameters of dataloader
all_dirs_path="/kaggle/working/CACD2000/*.jpg"
batch_size=16
num_train_data=1800
num_test_data=120
num_val_data=80
num_workers=2


# Parameters of model
gamma=0.08
lr=1e-4
max_epochs_vgg1=6
max_epochs_vgg2=6
max_epochs_em=5
max_epochs_em_sgd=5
cov=1
prior=0.95
loss_threshold=1e-6
weight_decay=1e-6
momentum=0.9


python train.py --batch-size $batch_size --num-train-data $num_train_data\
                --num-test-data $num_test_data --num-val-data $num_val_data \
                --num-workers $num_workers --gamma $num_workers --lr $lr \
                --max-epochs-vgg1 $max_epochs_vgg1 --max-epochs-vgg2 $max_epochs_vgg1\
                --max-epochs-em $max_epochs_em --max-epochs-em-sgd $max_epochs_em_sgd\
                --cov $cov --prior $prior --loss-threshold $loss_threshold\
                --weight-decay $weight_decay --momentum $momentum --all-dirs-path $all_dirs_path