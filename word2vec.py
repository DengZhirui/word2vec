import word2vec


def get_word2vec_dict(path):
    '''
    vocab_w = {} # {word1:1, word2:2}
    '''
    vocab_w = load_dict(path+'vocab_w.json')
    word2vec_dict = open(path+'word2vec_dict.txt', 'w')
    for word in vocab_w:
        word2vec_dict.write(word + '\n')
    word2vec_dict.close()


word2vec.word2vec(path_d+'word2vec_dict.txt', path_d+'word2vec.bin', size=config.wt_d_model, min_count=1, binary=True, verbose=True)
weight = word2vec.load(path_d + 'word2vec.bin').vectors
# send weight to nn.Embedding