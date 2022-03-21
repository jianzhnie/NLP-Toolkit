import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

if __name__ == '__main__':
    classifier = pipeline('sentiment-analysis')

    result = classifier('I hate you')[0]
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    result = classifier('I love you')[0]
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-cased-finetuned-mrpc')

    classes = ['not paraphrase', 'is paraphrase']

    sequence_0 = 'The company HuggingFace is based in New York City'
    sequence_1 = 'Apples are especially bad for your health'
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    # The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
    # the sequence, as well as compute the attention masks.
    paraphrase = tokenizer(sequence_0, sequence_2, return_tensors='pt')
    not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors='pt')

    paraphrase_classification_logits = model(**paraphrase).logits
    not_paraphrase_classification_logits = model(**not_paraphrase).logits

    paraphrase_results = torch.softmax(paraphrase_classification_logits,
                                       dim=1).tolist()[0]
    not_paraphrase_results = torch.softmax(
        not_paraphrase_classification_logits, dim=1).tolist()[0]

    # Should be paraphrase
    for i in range(len(classes)):
        print(f'{classes[i]}: {int(round(paraphrase_results[i] * 100))}%')

    # Should not be paraphrase
    for i in range(len(classes)):
        print(f'{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%')
