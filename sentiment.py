"""
Reads files from data/train.txt or dev.txt
and appends the sentence level sentiment to each word

say sentiment of <sentence_sentiment> is predicted on sentences level in train.txt
output in train_senti.txt is

<token1>\t<label1>\t<1_sentence_sentiment>\n
<token2>\t<label2>\t<1_sentence_sentiment>\n
<token3>\t<label3>\t<1_sentence_sentiment>\n
<token4>\t<label4>\t<1_sentence_sentiment>\n
\n
<token1>\t<label1>\t<2_sentence_sentiment>\n
<token2>\t<label2>\t<2_sentence_sentiment>\n
<token3>\t<label3>\t<2_sentence_sentiment>\n
<token4>\t<label4>\t<2_sentence_sentiment>\n
...

"""

from allennlp.predictors.predictor import Predictor
from allennlp.predictors.predictor import Predictor
import allennlp_models.classification
from time import time
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz")
print("LOADED ROBERTA LARGE!")
predictor._model = predictor._model.cuda()

tic = time()
p = predictor.predict(sentence="a very well-made, funny and entertaining picture.")
p2 = predictor.predict(sentence="a very well-made, funny and entertaining picture it was.")
toc = time()
print(p['probs'][0])  # Print positive sentiment
print("one eval takes", (toc-tic)/2, "seconds")

def read_data(path):
    X, Y, temp_tags, temp_line = [], [], [], []
    Sentiments = []

    num_sents = 0
    num_lines = 0
    a = open(path,'r').readlines()
    total_lines = len(a)
    for i in a:
        i = i.strip()

        if bool(i):
            temp_line.append(i.split()[0])
            temp_tags.append(i.split()[1])
        else:
            Sentiments.append(predictor.predict(sentence=(" ".join(temp_line)))['probs'][0])
            X.append(temp_line)
            Y.append(temp_tags)
            temp_line, temp_tags = [], []

            num_sents += 1
            if num_sents % 100 == 0:
                print(num_sents, "at lines =", num_lines, "/", total_lines)
                # print(Sentiments[-1], "for", " ".join(X[-1]))
        num_lines += 1
    print(num_sents," Sentences Read from", num_lines, "lines.")
    return X, Y, Sentiments


def save_data(X, Y, Sentiments, save_path):
    assert len(X) == len(Y)
    assert len(X) == len(Sentiments)

    fo = open(save_path, "w+")
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        sentence_level_sentiment = "%.10f" % Sentiments[i] # precision upto 10 decimal point

        assert len(x) == len(y)
        for j in range(len(x)):
            token = x[j]
            tag = y[j]

            line = token + "\t" + tag + "\t" + sentence_level_sentiment + "\n"
            fo.write(line)
        
        fo.write("\n")

        if i%500 == 0:
            print(i, "/", len(X), "Sentences written")

    fo.close()
    return

if __name__ == "__main__":
    train_path = "./data/train.txt"
    dev_path = "./data/dev.txt"

    train_new_path = "./data/train_senti.txt"
    dev_new_path = "./data/dev_senti.txt"

    X, Y, Sentiments = read_data(train_path)
    print("\n=============\nRead Training data and sentiment performed.\n=============\n")
    save_data(X, Y, Sentiments, train_new_path)
    print("\n=============\nSaved Training data with sentiment.\n=============\n")

    X, Y, Sentiments = read_data(dev_path)
    print("\n=============\nRead Training data and sentiment performed.\n=============\n")
    save_data(X, Y, Sentiments, dev_new_path)
    print("\n=============\nSaved Training data with sentiment.\n=============\n")
