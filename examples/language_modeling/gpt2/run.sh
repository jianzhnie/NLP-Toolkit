torchrun --nproc-per-node 1 fsdp_finetune.py \
    --data_root /home/robin/datasets/prompt_data/olcc/olcc_alpaca.json \
    --checkpoint gpt2 \
    --output-dir /home/robin/work_dir/llm/nlp-toolkit/work_dir/gpt2
