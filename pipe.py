# TODO
# 1. Group Classes
# 2. Token vs Sentence Level classification
# 3. Auxilliary Loss with options, masks for V 
# Last step: Integrate CRF for propaganda labels

import torch
from torch.utils import data
from transformers import BertTokenizer
from params import params
import os

from torch import nn
from torch.nn.functional import relu, tanh, sigmoid
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_bert import BertPreTrainedModel

import wandb
from early_stopping import EarlyStopping
from routines import train, eval
import random

train_path = "./data/train_senti_lex.txt"
dev_path = "./data/dev_senti_lex.txt"
torch.manual_seed(params.seed)
random.seed(params.seed)

def read_data(path, isTest=False):
    """
    Dataloader to read file in the format of a .txt file as
    train.txt\n
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
    Returns:
    X, Y, Sentiment, Lexicons
    Where X is the sentence list and Y is the non 'O' Ignore tag
    """

    temp_line = []
    temp_tags = []
    temp_lexi = []
    X = []
    Y = []
    Sentiment = []
    Lexicons = []
    count = 0

    a = open(path,'r').readlines()
    for i in a:
        i = i.strip()
        
        if bool(i):
            temp_line.append(i.split()[0])
            temp_tags.append(i.split()[1])
            temp_lexi.append(list(map(float, (i.split()[3:]))))
            senti = float(i.split()[2])
        else:
            if set(temp_tags) == {"O"}:
                count += 1
                if count < 100:                               ### Adding 100 samples of the O class
                    if params.sentence_level:
                        X.append(" ".join(temp_line))
                        Y.append("O")
                    else:
                        X.append(temp_line)
                        Y.append(temp_tags)
                    Sentiment.append(senti)
                    senti = None
                    Lexicons.append(temp_lexi)
            else:
                if params.sentence_level:
                    X.append(" ".join(temp_line))
                    Y.append( list(filter(("O").__ne__, temp_tags))[0] )
                else:
                    X.append(temp_line)
                    Y.append(temp_tags)

                Sentiment.append(senti)
                senti = None
                Lexicons.append(temp_lexi)

            temp_line = []
            temp_tags = []
            temp_lexi = []
    
    print(count," dropped")
    return (X, Y, Sentiment, Lexicons)

class PropDataset(data.Dataset):
    """
    Dataloader class\n
    Calls the read_data funciton\n

    """
    def __init__(self, path, isTest=False):
        
        (X, Y, Sentiment, Lexicons) = read_data(path)
        print(len(X), len(Y), len(Sentiment), len(Lexicons))
        print(X[:2], Y[:2], Sentiment[:2], Lexicons[:2])
        
        if params.dummy_run:
            X = X[:32]
            Y = Y[:32]
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        assert len(X) == len(Y)
        assert len(Sentiment) == len(X)
        assert len(Lexicons) == len(X)

        tag2idx = {}
        idx2tag = {}

        tag2idx["Red_Herring"] = 1  # 1 is classify and delete
        tag2idx["Name_Calling,Labeling"] = 1
        tag2idx["Reductio_ad_hitlerum"] = 1
        tag2idx["Repetition"] = 1  # Maybe Put to others class

        # 2 is the Style Transfer Class
        tag2idx["Obfuscation,Intentional_Vagueness,Confusion"] = 2
        tag2idx["Loaded_Language"] = 2

        tag2idx["Slogans"] = 0  # 0 is the ignore class 'O'
        tag2idx["Appeal_to_fear-prejudice"] = 0
        tag2idx["Doubt"] = 0
        tag2idx["Exaggeration,Minimisation"] = 0
        tag2idx["Flag-Waving"] = 0
        tag2idx["Bandwagon"] = 0
        tag2idx["Causal_Oversimplification"] = 0
        tag2idx["Appeal_to_Authority"] = 0
        tag2idx["Black-and-White_Fallacy"] = 0
        tag2idx["Thought-terminating_Cliches"] = 0
        tag2idx["Straw_Men"] = 0
        tag2idx["Whataboutism"] = 0

        if params.group_classes:
            tag2idx["O"] = 0
            tag2idx["CD"] = 1
            tag2idx["ST"] = 2
            tag2idx["<PAD>"] = 3

            idx2tag[0] = "O"
            idx2tag[1] = "CD"
            idx2tag[2] = "ST"
            idx2tag[3] = "<PAD>"
        else:
            labels = list(tag2idx.keys())
            tag2idx = {key:idx+1 for idx, key in enumerate(labels)}
            tag2idx["O"] = 0
            tag2idx["<PAD>"] = len(labels)+1
            idx2tag = {value:key for key, value in tag2idx.items()}

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag 

        self.sents , self.tags = X , Y
        self.sentiments, self.lexicons = Sentiment, Lexicons

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        words   = self.sents[index]
        # print(" ".join(words))
        tags    = self.tags[index]
        lexicon_sequence = self.lexicons[index]
        sentiment = self.sentiments[index]

        if params.sentence_level:
            tag_labels = [self.tag2idx[tags]]
            input_ids = self.tokenizer.encode(words , add_special_tokens=True , do_lower_case = False )
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"])
            tag_labels = [self.tag2idx["<PAD>"]]
            lexicons_subworded = [[0.5, 0.5, 0.5]]

            for w, t, l in zip(words, map(lambda x: self.tag2idx[x], tags), lexicon_sequence):
                tokens = self.tokenizer.tokenize(w)
                xx = self.tokenizer.convert_tokens_to_ids(tokens)

                is_head = [1] + [0] * (len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))

                tag_labels.extend([t] * len(tokens))
                lexicons_subworded.extend([l] * len(tokens))
                input_ids.extend(xx)

            input_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
            tag_labels.append(self.tag2idx["<PAD>"])
            lexicons_subworded.append([0.5, 0.5, 0.5])

            lexicon_sequence = lexicons_subworded

        seq_len = len(input_ids)

        if seq_len < 210:
            input_ids = input_ids+[100]*(210-seq_len)
            att_mask = [1]*seq_len+[0]*(210-seq_len)
        else:
            input_ids = input_ids[:210]
            att_mask = [1] * 210

        y = tag_labels
        sentiment = [sentiment]
    
        input_ids = torch.LongTensor(input_ids).to(params.device)
        att_mask = torch.Tensor(att_mask).to(params.device)
        sentiment = torch.Tensor(sentiment).to(params.device)

        if params.sentence_level:
            y = torch.LongTensor(y).to(params.device)
            return input_ids , y , att_mask, seq_len

        less_by = 210 - len(lexicon_sequence)
        y += [self.tag2idx["<PAD>"]] * less_by
        lexicon_sequence += [[0.5, 0.5, 0.5]] * less_by
        y = torch.LongTensor(y).to(params.device)
        lexicon_sequence = torch.Tensor(lexicon_sequence).to(params.device)

        return input_ids, y, att_mask, lexicon_sequence, sentiment, seq_len

# b = PropDataset(dev_path)
# x = b.__getitem__(0)
# print(x, "\n", list(map(lambda x: x.shape , x[:-1])), '\n', list(map(lambda x: x.type(), x[:-1])))
# exit()
"""
Model Class
"""
class BertMultiTaskLearning(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiTaskLearning, self).__init__(config)
        if params.group_classes:
            self.num_labels = 4
        else:
            self.num_labels = 20

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if params.sentence_level:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        else:
            self.bert_out_linear = nn.Linear(config.hidden_size, params.multitask_feat_dims)
            self.bert_out_non_linear = torch.nn.LeakyReLU()

            self.lexicon_linear = nn.Linear(params.multitask_feat_dims, 3)

            self.sentiment_att_linear = nn.Linear(params.multitask_feat_dims, 1)
            self.sentiment_linear = nn.Linear(config.hidden_size, 1)

            self.classifier = nn.Linear(params.multitask_feat_dims, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None, sentiments=None, lexicons=None):
        if params.sentence_level:
            # input_ids = input_ids.to(params.device)
            # attention_mask = attention_mask.to(params.device)
            # print("Shape is ",input_ids.shape)

            output = self.bert(input_ids, attention_mask=attention_mask)
        
            pooled_output = self.dropout(output[1])
            logits = self.classifier(pooled_output)

            # add hidden states and attention if they are here
            outputs = (logits,) + output[2:]

            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels) , labels.view(-1))

                outputs = (loss,) + outputs
            return outputs  # (loss), logits, (hidden_states), (attentions)
        else:
            outputs = self.bert(input_ids, attention_mask)
            outputs = outputs[0]
            linear_outputs = self.dropout(self.bert_out_non_linear(self.bert_out_linear(outputs)))

            # Lexicons
            lexical_predictions = torch.sigmoid(self.lexicon_linear(linear_outputs))

            # For Sentiment, attend to weight and add
            bool_masks = (attention_mask == 0)
            att_weights = self.sentiment_att_linear(linear_outputs)
            softmaxed_att_weights = torch.softmax(att_weights.masked_fill(bool_masks.unsqueeze(-1), -10000.0), 1)

            pooled_vectors = torch.sum(outputs * att_weights, axis = 1) # (batch_size, self.config.hidden)
            sentiments_guess = torch.sigmoid(self.sentiment_linear(pooled_vectors)) # (batch_size, 1)

            # Labels
            labels = self.classifier(linear_outputs)

            return labels, sentiments_guess, lexical_predictions

"""
The main Runnning 
"""
if __name__ == "__main__":
    if params.wandb:
        wandb.init(project="news_bias", name=params.run)
    
    model = BertMultiTaskLearning.from_pretrained('bert-base-uncased')
    print("Detected ", torch.cuda.device_count(), "GPUs!")

    model = nn.DataParallel(model)
    model.to(params.device) 

    train_dataset = PropDataset(train_path, False)
    eval_dataset = PropDataset(dev_path, True)

    train_iter = data.DataLoader(dataset=train_dataset, batch_size= params.batch_size, shuffle= True)
    eval_iter = data.DataLoader(dataset=eval_dataset, batch_size=params.batch_size, shuffle=False)

    warmup_proportion = 0.1
    num_train_optimization_steps = int(
        len(train_dataset) / params.batch_size) * params.n_epochs
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=params.lr,
    #                      warmup=warmup_proportion,
    #                      t_total=num_train_optimization_steps)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=params.lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        warmup_proportion*num_train_optimization_steps), num_training_steps=num_train_optimization_steps)

    ignore_index = 3 if params.group_classes else 19
    criterion = {'label_crit': torch.nn.CrossEntropyLoss(ignore_index=ignore_index , reduction='mean'),
                'sentiment_crit': torch.nn.MSELoss(reduction='mean'),
                'lexicon_crit': torch.nn.MSELoss(reduction='none')
               }

    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=params.patience, verbose=True)

    # Eval before beginning
    # _, _, _, _, _ = eval(model=model, iterator=eval_iter, criterion=criterion) 

    """
    Beginning of the Training Loop
    """
    for epoch in range(1,params.n_epochs):

        print("==========Running Epoch {}==========".format(epoch))

        # if not os.path.exists('checkpoints'):
        #     os.makedirs('checkpoints')
        # if not os.path.exists('results'):
        #     os.makedirs('results')
        # fname = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run)
        # spath = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run+".pt")

        train_loss = train(model, iterator=train_iter, optimizer=optimizer, scheduler=scheduler, criterion=criterion)

        precision, recall, f1, valid_loss, group_report = eval(model=model, iterator=eval_iter, criterion=criterion) 

        if params.group_classes:
            print("CD F1 is {}\n ST F1 is {}\n O F1 is {}\n".format(
                group_report["1"]["f1-score"], group_report["2"]["f1-score"], 
                group_report["0"]["f1-score"]))

            if params.wandb:
                wandb.log({"Training Loss": train_loss.item(), "Validation Loss": valid_loss.item(
                ), "Precision": precision, "Recall": recall, "F1": f1,
                "CD_F1": group_report["1"]["f1-score"], "ST_F1": group_report["2"]["f1-score"],
                "O_F1": group_report["0"]["f1-score"]})
        else:
            print(f"F1 = {f1:.5f} precision = {precision:.5f} recall = {recall:.5f}")

            train_dataset.tag2idx
            wandb_log = {train_dataset.idx2tag[i] + " F1": group_report[str(i)]['f1-score'] for i in range(19)}
            wandb_log["Precision"] = precision
            wandb_log["Recall"] = recall
            wandb_log["F1"] = f1
    
            if params.sentence_level:
                wandb_log["Training Loss"] = np.average(train_loss).item()
                wandb_log["Validation Loss"] = np.average(valid_loss).item()
            else:
                wandb_log["Training Loss"] = train_loss[0]
                wandb_log["Training Label Loss"] = train_loss[1]
                wandb_log["Training Sentiment Loss"] = train_loss[2]
                wandb_log["Training Lexicon Loss"] = train_loss[3]

                wandb_log["Validation Loss"] = valid_loss[0]
                wandb_log["Validation Label Loss"] = valid_loss[1]
                wandb_log["Validation Sentiment Loss"] = valid_loss[2]
                wandb_log["Validation Lexicon Loss"] = valid_loss[3]

            if params.wandb:
                wandb.log(wandb_log)


        epoch_len = len(str(params.n_epochs))
        if params.sentence_level:
            print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        else:
            print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                     f'train_loss: {train_loss} ' +
                     f'valid_loss: {valid_loss}')

        print(print_msg)

        # early_stopping(-1*f1, model, spath)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
