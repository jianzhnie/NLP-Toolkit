from transformers import pipeline

if __name__ == '__main__':
    text_generator = pipeline('text-generation')
    print(
        text_generator('As far as I am concerned, I will',
                       max_length=500,
                       do_sample=False))
