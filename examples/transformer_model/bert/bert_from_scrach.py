from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from transformers import (AlbertConfig, AlbertForMaskedLM, BertTokenizerFast,
                          DataCollatorForLanguageModeling,
                          LineByLineTextDataset, Trainer, TrainingArguments)

if __name__ == '__main__':
    files = './lunyu.txt'  # 训练文本文件
    vocab_size = 10000
    min_frequency = 2
    limit_alphabet = 10000
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]',
                      '[MASK]']  # 适用于Bert和Albert

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True,
    )

    # Customize training
    tokenizer.train(files,
                    vocab_size=vocab_size,
                    min_frequency=min_frequency,
                    show_progress=True,
                    special_tokens=special_tokens,
                    limit_alphabet=limit_alphabet,
                    wordpieces_prefix='##')

    tokenizer = BertWordPieceTokenizer('./tokenizer/vocab.txt', )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ('[CLS]', tokenizer.token_to_id('[SEP]')),
        ('[SEP]', tokenizer.token_to_id('[CLS]')),
    )
    tokenizer.enable_truncation(max_length=512)

    config = AlbertConfig(
        vocab_size=1359,
        embedding_size=256,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )

    tokenizer = BertTokenizerFast.from_pretrained('./tokenizer',
                                                  padding=True,
                                                  truncation=True,
                                                  max_len=512)
    model = AlbertForMaskedLM(config=config)
    model.num_parameters()
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='./lunyu.txt',
        block_size=256,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir='./lunyuAlbert',
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_gpu_train_batch_size=16,
        save_steps=2000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )
