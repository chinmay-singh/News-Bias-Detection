"""
Reads files from data/train_senti.txt or dev/train_senti.txt
and appends the sentence level sentiment to each word

say sentiment of <sentence_sentiment> is predicted on sentences level in train_senti.txt
output in train_senti_lex.txt is

<token1>\t<label1>\t<1_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
<token2>\t<label2>\t<1_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
<token3>\t<label3>\t<1_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
<token4>\t<label4>\t<1_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
\n
<token1>\t<label1>\t<2_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
<token2>\t<label2>\t<2_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
<token3>\t<label3>\t<2_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
<token4>\t<label4>\t<2_sentence_sentiment>\t<valence1>\t<dominance1>\t<arousal1>\n
...

"""
lexi_path = "lexicon_NRC/NRC-VAD-Lexicon.txt"
fo = open(lexi_path, "r")
lines = list(map(lambda x: x.strip().split(), fo.readlines()[1:-1]))
token_to_values_map = {(" ".join(line[0:-3])).strip().lower(): line[-3:] for line in lines}
print(list(map(type, lines[0])))
print(list(token_to_values_map.keys())[:100])
fo.close()

# Also map to lexicons
def read_data(path):
    X, Y, temp_tags, temp_line = [], [], [], []
    Sentiments, Lexical_Attributes = [], []

    num_sents = 0
    num_lines = 0
    a = open(path,'r').readlines()
    total_lines = len(a)
    for i in a:
        i = i.strip()

        if bool(i):
            temp_line.append(i.split()[0])
            temp_tags.append(i.split()[1])
            senti = i.split()[2]
        else:
            Sentiments.append(senti)
            senti = None

            X.append(temp_line)
            Y.append(temp_tags)

            temp_lexi_attr = []
            
            idx = 0
            while idx < (len(temp_line) - 2):
                trigram = " ".join(temp_line[idx:idx+3]).strip().lower()
                if trigram in token_to_values_map.keys():
                    temp_lexi_attr.append(token_to_values_map[trigram])
                    temp_lexi_attr.append(token_to_values_map[trigram])
                    temp_lexi_attr.append(token_to_values_map[trigram])
                    idx += 3
                else:
                    bigram = " ".join(temp_line[idx:idx+2]).strip().lower()
                    if bigram in token_to_values_map.keys():
                        temp_lexi_attr.append(token_to_values_map[bigram])
                        temp_lexi_attr.append(token_to_values_map[bigram])
                        idx += 2
                    else:
                        temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                        idx += 1

            if len(temp_line) > 2 and idx != len(temp_line) - 2:
                print(idx, len(temp_line), temp_line)

            if (len(temp_line)-idx) == 2:
                bigram = " ".join(temp_line[idx:idx+2]).strip().lower()
                if bigram in token_to_values_map.keys():
                    temp_lexi_attr.append(token_to_values_map[bigram])
                    temp_lexi_attr.append(token_to_values_map[bigram])
                    idx += 2
                else:
                    temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                    idx += 1
                    temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                    idx += 1
            else:
                temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                idx += 1
            
            assert idx == len(temp_line)

            Lexical_Attributes.append(temp_lexi_attr)
            temp_line, temp_tags = [], []

            num_sents += 1
            if num_sents % 100 == 0:
                print(num_sents, "at lines =", num_lines, "/", total_lines)
        num_lines += 1
    print(num_sents," Sentences Read from", num_lines, "lines.")
    return X, Y, Sentiments, Lexical_Attributes


def save_data(X, Y, Sentiments, Lexical_Attributes, save_path):
    assert len(X) == len(Y)
    assert len(X) == len(Sentiments)
    assert len(X) == len(Lexical_Attributes)

    fo = open(save_path, "w+")
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        sentence_level_sentiment = Sentiments[i]
        lexical_atts = Lexical_Attributes[i]

        assert len(x) == len(y)
        assert len(lexical_atts) == len(x)
        for j in range(len(x)):
            token = x[j]
            tag = y[j]
            valence, dominance, arousal = lexical_atts[j]

            line = token + "\t" + tag + "\t" + sentence_level_sentiment + \
                    "\t" + valence + "\t" + dominance + "\t" + arousal + "\n"
            fo.write(line)
        
        fo.write("\n")

        if i%500 == 0:
            print(i, "/", len(X), "Sentences written")

    fo.close()
    return

if __name__ == "__main__":
    train_path = "./data/train_senti.txt"
    dev_path = "./data/dev_senti.txt"

    train_new_path = "./data/train_senti_lex.txt"
    dev_new_path = "./data/dev_senti_lex.txt"

    X, Y, Sentiments, Lexical_Attributes = read_data(train_path)
    print("\n=============\nRead Training data and sentiment performed.\n=============\n")
    save_data(X, Y, Sentiments, Lexical_Attributes, train_new_path)
    print("\n=============\nSaved Training data with sentiment.\n=============\n")

    X, Y, Sentiments, Lexical_Attributes = read_data(dev_path)
    print("\n=============\nRead Training data and sentiment performed.\n=============\n")
    save_data(X, Y, Sentiments, Lexical_Attributes, dev_new_path)
    print("\n=============\nSaved Training data with sentiment.\n=============\n")
