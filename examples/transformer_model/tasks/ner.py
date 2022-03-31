from transformers import pipeline

if __name__ == '__main__':
    ner_pipe = pipeline('ner')

    sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO,
    therefore very close to the Manhattan Bridge which is visible from the window."""

    for entity in ner_pipe(sequence):
        print(entity)
