CUDA_VISIBLE_DEVICES=1 python main.py with exp_name="solo_clss" mode='train' dataset_name='nuswide81' \
                    topK=5000 machine_name='3080' n_class=81 \
                    eval_interval=20  epoch=61 \
                    n_book=4

CUDA_VISIBLE_DEVICES=1 python main.py with exp_name="solo_clss" mode='train' dataset_name='nuswide81' \
                    topK=5000 machine_name='3080' n_class=81 \
                    eval_interval=20  epoch=61 \
                    n_book=8

CUDA_VISIBLE_DEVICES=1 python main.py with exp_name="solo_clss" mode='train' dataset_name='nuswide81' \
                    topK=5000 machine_name='3080' n_class=81 \
                    eval_interval=20  epoch=61 \
                    n_book=12

CUDA_VISIBLE_DEVICES=1 python main.py with exp_name="solo_clss" mode='train' dataset_name='nuswide81' \
                    topK=5000 machine_name='3080' n_class=81 \
                    eval_interval=20  epoch=61 \
                    n_book=16
