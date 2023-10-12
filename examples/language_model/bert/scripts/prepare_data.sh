python create_pretraining_data.py \
  --input_file=/home/robin/work_dir/llm/nlp-toolkit/text_data/ptb \
  --vocab_file=/home/robin/work_dir/llm/nlp-toolkit/text_data/ptb/vocab/vocab \
  --output_file=/home/robin/work_dir/llm/nlp-toolkit/text_data/ptb/training_data.hdf5 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
