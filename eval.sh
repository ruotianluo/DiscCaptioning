#! /bin/sh
id=$1

python eval.py --decoding_constraint 1 --dump_images 0 --num_images -1 --batch_size 50 --split $2  --input_label_h5 data/cocotalk_label.h5 --input_fc_dir data/cocotalk_fc --input_att_dir data/cocobu_att --model log_$id/model.pth --language_eval 1 --beam_size 5 --temperature 1.0 --sample_max 1 --infos_path log_$id/infos_$id.pkl
