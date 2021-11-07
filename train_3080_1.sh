CUDA_VISIBLE_DEVICES=0 python main.py with exp_name="solo_clss" mode='train' dataset_name='imagenet' \
                    topK=1000 machine_name='3080' n_class=100  \
                    eval_interval=20  epoch=61 \
                    n_book=4

CUDA_VISIBLE_DEVICES=0 python main.py with exp_name="solo_clss" mode='train' dataset_name='imagenet' \
                    topK=1000 machine_name='3080' n_class=100 \
                    eval_interval=20  epoch=61 \
                    n_book=8

CUDA_VISIBLE_DEVICES=0 python main.py with exp_name="solo_clss" mode='train' dataset_name='imagenet' \
                    topK=1000 machine_name='3080' n_class=100 \
                    eval_interval=20  epoch=61 \
                    n_book=12

CUDA_VISIBLE_DEVICES=0 python main.py with exp_name="solo_clss" mode='train' dataset_name='imagenet' \
                    topK=1000 machine_name='3080' n_class=100 \
                    eval_interval=20  epoch=61 \
                    n_book=16

