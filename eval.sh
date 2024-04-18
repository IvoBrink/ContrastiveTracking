# Eval on validation dataset



# 1600 iters
python test.py tracking --exp_id mot17_with_contrastive --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../exp/tracking/mot17_with_contrastive/model_5.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --max_age 32
#python test.py tracking --exp_id mot17_with_contrastive --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../exp/tracking/mot17_with_contrastive.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --max_age 32 --train
python test.py tracking --exp_id mot17_without_contrastive --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../exp/tracking/mot17_without_contrastive/model_5.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --max_age 32

# 6400 iters
python test.py tracking --exp_id mot17_with_contrastive_6400 --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../exp/tracking/mot17_with_contrastive_6400/model_last.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --public_det --load_results ../data/mot17/results/val_half_det.json
python test.py tracking --exp_id mot17_without_contrastive_6400 --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../exp/tracking/mot17_without_contrastive_6400/model_last.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --public_det --load_results ../data/mot17/results/val_half_det.json