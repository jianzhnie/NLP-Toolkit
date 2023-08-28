train_text = [
    'just plain boring',
    'entirely predictable and lacks energy',
    'no surprises and very few laughs',
    'very powerful',
    'the most fun film of the summer',
]
labels = [0, 0, 0, 1, 1]
test_text = ['predictable with no fun']

num_vocab = len(set(' '.join(train_text).split()))
print('num_vocab:', num_vocab)
num_train_vocab = len(set(' '.join(train_text[:3]).split()))
print('num_train_vocab:', num_train_vocab)
num_test_vocab = len(set(' '.join(train_text[3:]).split()))
print('num_test_vocab:', num_test_vocab)
