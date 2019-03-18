import numpy as np
import re
import collections
import io

stopwords = []

def read_data(_file, cleaning):
    revs = []
    max_len = 0
    words_list = []
    with io.open(_file, "r", encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            ID, label, sentence = line.split('\t')
            label_idx = 1 if label == 'pos' else 0  # 1 for pos and 0 for neg
            rev = []
            rev.append(sentence.strip())

            if cleaning:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            revs.append({'y': label_idx, 'txt': orig_rev})
            words_list += orig_rev.split()
    return revs, words_list

contractions_dict = {
    "won't": "were not",
    "you'll": "you will",
    "we're": "we are",
    "that's": "that is",
    "were't": "were not",
    "i'd": "i do not",
    "i'll": "i will",
    "there's": "there is",
    "they'll": "they will",
    "it's": "it is",
    "they're": "they are",
    "i've": "i have",
    "we'll": "we will",
    "she's": "she is",
    "could": "could have",
    "we've": "we have",
    "you'd": "you don't",
    "you're": "you are",
    "they've": "they have",
    "shouldn't": "should not",
    "he's": "he is ",
    "should ve": "should have",
    "could've": "could have",
    "couldn't've": "could not have",
    "did n't": "did not",
    "do n't": "do not",
    "had n't": "had not",
    "had n't've": "had not have",
    "has n't": "has not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "should've": "should have",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "there'd": "here would",
    "there'd've": "there would have",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll've": "they will have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll've": "we will have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd've": "you would have",
    "you'll've": "you will have",
    "you've": "you have",
    "n't": "not",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "isn't": "is not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "i'm": "i am",
}

def clean_str(string):
    """
    TODO: Data cleaning
    """
    string = string.lower()
    string = string.strip()
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, s)

    string = expand_contractions(string)
    string = re.sub(r'[^A-Za-z]', ' ', string)
    # 3- Remove Stop words
    words = string.split()
    new_string = [word for word in words if word not in stopwords]
    return ' '.join(new_string)


def build_vocab(words_list, max_vocab_size=-1):
    """
    TODO: 
        Build a word dictionary, use max_vocab_size to limit the total number of vocabulary.
        if max_vocab_size==-1, then use all possible words as vocabulary. The dictionary should look like:
        ex:
            word2idx = { 'UNK': 0, 'i': 1, 'love': 2, 'nlp': 3, ... }
        
        top_10_words is a list of the top 10 frequently words appeared
        ex:
            top_10_words = ['a','b','c','d','e','f','g','h','i','j']
    """
    # word2idx = {'UNK': 0} # UNK is for unknown word
    word2idx = {}
    counter = collections.Counter(words_list)
    if max_vocab_size == -1:
        word2idx.update(counter)
    else:
        word2idx.update(counter.most_common(max_vocab_size))
    top_10_words = []

    for key, _ in counter.most_common(10):
        top_10_words.append(key)
    return word2idx, top_10_words


def get_info(revs, words_list):
    """
    TODO: 
        First check what is revs. Then calculate max len among the sentences and the number of the total words
        in the data.
        nb_sent, max_len, word_count are scalars
    """
    nb_sent, max_len, word_count = 0, 0, 0
    nb_sent = len(revs)

    # find the largest length of the sentence
    for sen in revs:
        if len(sen['txt']) > max_len:
            max_len = len(sen['txt'])

    word_count = len(words_list)
    return nb_sent, max_len, word_count


def data_preprocess(_file, cleaning, max_vocab_size):
    revs, words_list = read_data(_file, cleaning)
    nb_sent, max_len, word_count = get_info(revs, words_list)
    word2idx, top_10_words = build_vocab(words_list, max_vocab_size)
    # data analysis
    print("Number of words: ", word_count)
    print("Max sentence length: ", max_len)
    print("Number of sentences: ", nb_sent)
    print("Number of vocabulary: ", len(word2idx))
    print("Top 10 most frequently words", top_10_words)

    return revs, word2idx


def feature_extraction_bow(revs, word2idx):
    """
    TODO: 
        Convert sentences into vectors using BoW. 
        data is a 2-D array with the size (nb_sentence*nb_vocab)
        label is a 2-D array with the size (nb_sentence*1)
    """
    # BOW
    vocab = list(word2idx.keys())
    data = np.zeros((len(revs), len(vocab)))
    i = 0
    for sent in revs:
        sent_word = sent['txt'].split()
        for word in sent_word:
            if word in vocab:
                data[i, vocab.index(word)] += 1
        i += 1
    '''
    // Bi-Gram (Comment the upper BOW part and Uncomment the following to use Bi-gram)
    tupleList = []
    data = np.zeros((len(word2idx), len(word2idx)))
    allSentences = [sentence["txt"] for sentence in revs]
    for sentence in allSentences:
        words = clean_str(sentence).split()
        # print (words)
        for index, word in enumerate(words):
            # print (index, len(words))
            if (index + 1) != len(words):
                tupleList.append(tuple([word, words[index + 1]]))
                for i, key in enumerate(word2idx):
                    # print (i , key, word)
                    if (key == word):
                        for j, key2 in enumerate(word2idx):
                            # print (i, key2, words[index+1])
                            if (key2 == words[index + 1]):
                                data[i][j] += 1
    '''
    label = []
    for sent_info in revs:
        label.append([sent_info['y']])
    return np.array(data), np.array(label)


def normalization(data):
    """
    TODO: 
        Normalize each dimension of the data into mean=0 and std=1
    """
    return (data - data.mean(axis=0)) / data.std(axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f', '--file', help='input csv file', required=False, default='./twitter-sentiment.csv')
    parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv', '--max_vocab', help='max vocab size predifined, no limit if set -1', required=False,
                        default=-1)
    args = vars(parser.parse_args())
    print(args)

    revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))
    data, label = feature_extraction_bow(revs, word2idx)
    data = normalization(data)
