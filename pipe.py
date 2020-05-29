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


train_path = "./data/train.txt"
dev_path = "./data/dev.txt"



def read_data(path, isTest=False):
    """
    Dataloader to read file in the format of a .txt file as
    train.txt\n
    --word\\ttag\\n   \n
    --word\\ttag\\n     Sentence 1 \n 
    --word\\ttag\\n   \n
    \\t\\n            \n  
    --word\\ttag\\n   \n
    --word\\ttag\\n      Sentence 2  \n 
    --word\\ttag\\n   \n
    \n
    Returns:
    X, Y
    Where X is the sentence list and Y is the non 'O' Ignore tag
    """
    
    

    temp_line = []
    temp_tags = []
    X = []
    Y = []
    count = 0
    
    a = open(path,'r').readlines()
    
    for i in a:

        i = i.strip()
        
        if bool(i):
            temp_line.append(i.split()[0])
            temp_tags.append(i.split()[1])
        
        
        else:
            if set(temp_tags) == {"O"}:
                count += 1
                continue
            
            X.append(" ".join(temp_line))


            Y.append( list(filter(("O").__ne__, temp_tags))[0] )

            # print(temp_tags)

            temp_line = []
            temp_tags = []
    
    print(count," dropped")
    
    return (X, Y)

# a = read_data(dev_path)
# print(a[0][5])
# print(a[1][5])


class PropDataset(data.Dataset):

    """
    Dataloader class\n
    Calls the read_data funciton\n

    """


    def __init__(self, path , isTest=False):
        
        X,Y = read_data(path)

        if params.dummy_run:
            X = [X[0]]
            Y = [Y[0]]
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        assert len(X) == len(Y)

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

        tag2idx["CD"] = 1
        tag2idx["O"] = 0
        tag2idx["ST"] = 2
        # tag2idx["<PAD>"] = 0

        idx2tag[0] = "O"
        idx2tag[1] = "CD"
        idx2tag[2] = "ST"
        # idx2tag[0] = "<PAD>"

        self.tag2dx = tag2idx
        self.idx2tag = idx2tag 


        self.sents , self.tags = X , Y

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):

        words   = self.sents[index]
        tag    = self.tags[index]
        tag_label = self.tag2dx[ tag ]

        input_ids = self.tokenizer.encode(
                    words , add_special_tokens=True , do_lower_case = False )

        if len(input_ids)>210:
            input_ids = input_ids[:210]

        y = [tag_label] 

        seq_len = len(input_ids)

        att_mask = [1]*seq_len

        return input_ids , y , att_mask , seq_len

# b = PropDataset(dev_path)
# print(b.__getitem__(0))

def pad(batch):

    def f(x): return [sample[x] for sample in batch] #access the Xth index element of the sample in this batch

    seq_len = f(-1)
    max_len = 210
    y = f(1)

    def f(x, seqlen): return [
        sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]

    input_ids = torch.LongTensor(f(0,max_len)).to(params.device)

    y = torch.LongTensor(y).to(params.device)

    att_mask = torch.Tensor(f(2,max_len)).to(params.device)

    return input_ids , y, att_mask , seq_len

"""
Model Class
"""


class BertMultiTaskLearning(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiTaskLearning, self).__init__(config)
        self.num_labels = 3
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear( config.hidden_size, 3 )
        self.init_weights()
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        # input_ids = input_ids.to(params.device)

        # attention_mask = attention_mask.to(params.device)

        output = self.bert(
                    input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
    
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
                loss = loss_fct(logits.view(-1, self.num_labels) , labels.view(-1) )

            outputs = (loss,) + outputs



        return outputs  # (loss), logits, (hidden_states), (attentions)







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

    train_dataset   =   PropDataset(train_path, False)
    eval_dataset    =   PropDataset(dev_path, True)

    train_iter      =   data.DataLoader(dataset=train_dataset,
                                        batch_size= params.batch_size,
                                        shuffle= True,
                                        collate_fn=pad)
    
    eval_iter       =   data.DataLoader(dataset=eval_dataset,
                                batch_size=params.batch_size,
                                shuffle=False,
                                collate_fn=pad)

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


    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=params.patience, verbose=True)




    """
    Beggining of the Training Loop
    """

    for epoch in range(1,params.n_epochs):

        print("==========Running Epoch {}==========".format(epoch))

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('results'):
            os.makedirs('results')
        fname = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run)
        spath = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run+".pt")

        train_loss  =   train(model, iterator=train_iter, optimizer = optimizer, 
                        scheduler=scheduler)


        avg_train_losses.append(train_loss.item())

        precision, recall, f1, valid_loss, group_report = eval(model=model, iterator=eval_iter) 

        # idx2tag[1] = "O"
        # idx2tag[2] = "CD"
        # idx2tag[3] = "ST"
        # idx2tag[0] = "<PAD>"



        if params.wandb:
            wandb.log({"Training Loss": train_loss.item(), "Validation Loss": valid_loss.item(
            ), "Precision": precision, "Recall": recall, "F1": f1,
             "CD_F1": group_report["2"]["f1-score"], "ST_F1": group_report["3"]["f1-score"],
            "O_F1": group_report["1"]["f1-score"]})

        
        epoch_len = len(str(params.n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        early_stopping(-1*f1, model, spath)

        if early_stopping.early_stop:
            print("Early stopping")
            break
