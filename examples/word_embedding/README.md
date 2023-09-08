# Word-level Language Modeling using RNN and Transformer

This example trains a multi-layer RNN (Elman, GRU, or LSTM) or Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python train_transformer.py --cuda --epochs 6 --model Transformer --lr 5
                                           # Train a Transformer model on Wikitext-2 with CUDA.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`) or Transformer module (`nn.TransformerEncoder` and `nn.TransformerEncoderLayer`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```shell
PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --d_model D_MODEL     size of word embeddings
  --num_hidden_size NUM_HIDDEN_SIZE
                        number of hidden units per layer
  --num_layers NUM_LAYERS
                        number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --num_heads NUM_HEADS
                        the number of heads in the encoder of the transformer model
```
