PATH_DIR=/volume/pt-train/users/wzhang/sdx-workspace/output/ori/baichuan-inc/Baichuan2-13B-Chat-artificial-data

nohup llamafactory-cli train \
    --do_predict \
    --model_name_or_path /volume/pt-train/users/wzhang/sdx-workspace/models/baichuan-inc/Baichuan2-13B-Chat \
    --eval_dataset mmt_test_artificial_no_think \
    --dataset_dir ./data \
    --template baichuan2 \
    --output_dir ${PATH_DIR}/think \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 12000 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 5000 \
    --predict_with_generate \
    --trust_remote_code True \
    --max_new_tokens 4096 \
    --enable_thinking True > ${PATH_DIR}/think.log 2>&1 &