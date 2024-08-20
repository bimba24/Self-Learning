# # base
# echo "RI-Running base block - Model: declare-lab/flan-alpaca-base, Rationale Generation - QCM-E"
# CUDA_VISIBLE_DEVICES=0 python main_central.py \
#     --data_root data/ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-base \
#     --user_msg rationale --img_type vit \
#     --bs 2 --eval_bs 2 --epoch 3 --lr 8e-5 --output_len 64 \
#     --use_caption --use_generate --final_eval --prompt_format QCM-E \
#     --output_dir experiments \
#     --evaluate_dir models/mm-cot-base-rationale

# echo "RI-Running base block - Model: declare-lab/flan-alpaca-base, Rationale Generation - QCMG-A"
# CUDA_VISIBLE_DEVICES=0 python main_central.py \
#     --data_root data/ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-base \
#     --user_msg rationale --img_type vit \
#     --bs 2 --eval_bs 2 --epoch 3 --lr 8e-5 --output_len 64 \
#     --use_caption --use_generate --prompt_format QCMG-A \
#     --output_dir experiments \
#     --eval_le models/mm-cot-base-rationale/predictions_ans_eval.json \
#     --test_le models/mm-cot-base-rationale/predictions_ans_test.json \
#     --evaluate_dir models/mm-cot-base-answer

# # large
# # rationale generation
# echo "RI-Running large block - Model: declare-lab/flan-alpaca-large, Rationale Generation - QCM-E"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#     --data_root data/ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-large \
#     --user_msg rationale --img_type vit \
#     --bs 2 --eval_bs 2 --epoch 3 --lr 5e-5 --output_len 64 \
#     --use_caption --use_generate --prompt_format QCM-E \
#     --output_dir experiments \
#     --evaluate_dir models/mm-cot-large-rationale

# # answer inference
# echo "RI-Running large block - Model: declare-lab/flan-alpaca-large, Answer Inference - QCMG-A"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main_central.py \
#     --data_root data/ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-large \
#     --user_msg answer --img_type vit \
#     --bs 2 --eval_bs 2 --epoch 3 --lr 5e-5 --output_len 64 \
#     --use_caption --use_generate --prompt_format QCMG-A \
#     --output_dir experiments \
#     --eval_le models/mm-cot-large-rationale/predictions_ans_eval.json \
#     --test_le models/mm-cot-large-rationale/predictions_ans_test.json \
#     --evaluate_dir models/mm-cot-large-answer

# rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 2  --epoch 3 --lr 5e-5 --output_len 64 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments
    --evaluate_dir models/mm-cot-large-rationale

# answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_central.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg answer --img_type vit \
    --bs 2 --eval_bs 2 --epoch 3 --lr 5e-5 --output_len 64  \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir experiments \
    --eval_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_eval.json \
    --test_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_test.json \
    --evaluate_dir models/mm-cot-large-answer