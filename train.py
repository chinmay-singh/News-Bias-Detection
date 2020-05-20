import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
from params import params
import numpy as np
from model import BertMultiTaskLearning

from transformers import BertConfig , AdamW, get_linear_schedule_with_warmup

# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from dataloader import PropDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag, num_task, masking
import time
from early_stopping import EarlyStopping
import wandb
import sklearn

import torch
import gc
count = 0
def check_cuda():
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(count)
                print(type(obj), obj.size())
                count+=1
        except:
            pass


timestr = time.strftime("%Y%m%d-%H%M%S")
# torch.set_default_tensor_type("torch.cuda.FloatTensor")
# if torch.cuda.is_available():
#     dev = "cuda:0"
# else:
# 	dev = "cpu"
if not params.run:
    params.run = timestr
if params.dummy_run:
    params.batch_size = 1

dev = params.device

def train(model, iterator, optimizer, criterion, binary_criterion):

    check_cuda()
    model.train()
    # print(model)
    check_cuda()
    train_losses = []

    for k, batch in enumerate(iterator):
        words, x, is_heads, att_mask, tags, y, seqlens = batch
        
        # words.to("cuda")
        # x = x.to(dev)
        # print("x = {}".format(x))
        check_cuda()
        # is_heads.to("cuda")
        # tags.to("cuda")
        # y.to("cuda")
        # seqlens.to("cuda")
        check_cuda()

        att_mask = torch.Tensor(att_mask)
        # att_mask = att_mask.to(dev)

        check_cuda()


        optimizer.zero_grad()
        logits, _ = model(x, attention_mask=att_mask)
                                #, 
        loss = []

        if num_task == 2:
            for i in range(num_task):
                logits[i] = logits[i].view(-1, logits[i].shape[-1])
            
            # check_cuda()

            y[0] = y[0].view(-1).to(dev)
            y[1] = y[1].float().to(dev)
            
            # check_cuda()

            loss.append(criterion(logits[0], y[0]))
            loss.append(binary_criterion(logits[1], y[1]))
        else:
            for i in range(num_task):
                check_cuda()
                logits[i] = logits[i].view(-1, logits[i].shape[-1])
                y[i] = y[i].view(-1).to(dev)
                loss.append(criterion(logits[i], y[i]))
        

        if num_task == 1:
            joint_loss = loss[0]
        elif num_task==2:
            joint_loss = params.alpha*loss[0] + (1-params.alpha)*loss[1]
        check_cuda()
        joint_loss.backward()

        optimizer.step()
        scheduler.step()

        train_losses.append(joint_loss.item())

        if k%100 == 0:
            print("step: {}, loss: {}".format(k,loss[0].item()))

    train_loss = np.average(train_losses)

    return train_loss

def eval(model, iterator, f, criterion, binary_criterion):
    model.eval()

    valid_losses = []


    Words, Is_heads = [], []
    Tags = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    Y_hats = [[] for _ in range(num_task)]
    with torch.no_grad():
        for _ , batch in enumerate(iterator):
            words, x, is_heads, att_mask, tags, y, seqlens = batch
            att_mask = torch.Tensor(att_mask)
            logits, y_hats = model(x, attention_mask=att_mask) # logits: (N, T, VOCAB), y: (N, T)
    
            loss = []
            if num_task == 2 or masking:
                for i in range(num_task):
                    logits[i] = logits[i].view(-1, logits[i].shape[-1]) # (N*T, 2)
                y[0] = y[0].view(-1).to(dev)
                y[1] = y[1].float().to(dev)
                loss.append(criterion(logits[0], y[0]))
                loss.append(binary_criterion(logits[1], y[1]))
            else:
                for i in range(num_task):
                    logits[i] = logits[i].view(-1, logits[i].shape[-1]) # (N*T, 2)
                    y[i] = y[i].view(-1).to(dev)
                    loss.append(criterion(logits[i], y[i]))

            if num_task == 1:
                joint_loss = loss[0]
            elif num_task == 2:
                joint_loss = params.alpha*loss[0] + (1-params.alpha)*loss[1]

            valid_losses.append(joint_loss.item())
            Words.extend(words)
            Is_heads.extend(is_heads)

            for i in range(num_task):
                Tags[i].extend(tags[i])
                Y[i].extend(y[i].cpu().numpy().tolist())
                Y_hats[i].extend(y_hats[i].cpu().numpy().tolist())
    valid_loss = np.average(valid_losses) 

    with open(f, 'w') as fout:
        y_hats, preds = [[] for _ in range(num_task)], [[] for _ in range(num_task)]
        if num_task == 1:
            for words, is_heads, tags[0], y_hats[0] in zip(Words, Is_heads, *Tags, *Y_hats):
                for i in range(num_task):
                    y_hats[i] = [hat for head, hat in zip(is_heads, y_hats[i]) if head == 1]
                    preds[i] = [idx2tag[i][hat] for hat in y_hats[i]]
                fout.write(words.split()[0])
                fout.write("\n")
                for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                    fout.write("{},{},{}\n".format(w,idx2tag[0][tag2idx[0][t1]],p_1))
                fout.write("\n")
        
        elif num_task == 2:
            false_neg = 0
            false_pos = 0
            true_neg = 0
            true_pos = 0
            for words, is_heads, tags[0], tags[1], y_hats[0], y_hats[1] in zip(Words, Is_heads, *Tags, *Y_hats):
                y_hats[0] = [hat for head, hat in zip(is_heads, y_hats[0]) if head == 1]
                preds[0] = [idx2tag[0][hat] for hat in y_hats[0]]
                preds[1] = idx2tag[1][y_hats[1]]
            
                if tags[1].split()[1:-1][0] == 'Non-prop' and preds[1] == 'Non-prop':
                    true_neg += 1
                elif tags[1].split()[1:-1][0] == 'Non-prop' and preds[1] == 'Prop':
                    false_pos += 1
                elif tags[1].split()[1:-1][0] == 'Prop' and preds[1] == 'Prop':
                    true_pos += 1
                elif tags[1].split()[1:-1][0] == 'Prop' and preds[1] == 'Non-prop':
                    false_neg += 1
                
                fout.write(words.split()[0])
                fout.write("\n")
                for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                    fout.write("{} {} {} {} {}\n".format(w,t1,tags[1].split()[1:-1][0],p_1,preds[1]))
                fout.write("\n")
            try:
                precision = true_pos / (true_pos + false_pos)
            except ZeroDivisionError:
                precision = 1.0
            try:
                recall = true_pos / (true_pos + false_neg)
            except ZeroDivisionError:
                recall = 1.0
            try:
                f1 = 2 *(precision*recall) / (precision + recall)
            except ZeroDivisionError:
                if precision*recall==0:
                    f1=1.0
                else:
                    f1=0.0
            print("sen_pre", precision)
            print("sen_rec", recall)
            print("sen_f1", f1)
            false_neg = false_pos = true_neg = true_pos = precision = recall = f1 = 0

    ## calc metric 
    y_true, y_pred = [], []

    '''
    SOme error in the writing of file is being made debug this later
    '''

    for i in range(num_task):
        y_true.append(np.array([tag2idx[i][line.split(',')[i+1].strip()] for line in open(
            f, 'r').read().splitlines() if (len(line.split(',')) > 1) and (line.split(',')[i+1].strip() in tag2idx[0].keys()) and (line.split(',')[i+1+num_task].strip() in tag2idx[0].keys())]))
        
        y_pred.append(np.array([tag2idx[i][line.split(',')[i+1+num_task].strip()] for line in open(f, 'r').read().splitlines() if len(line.split(',')) > 1 and (line.split(',')[i+1].strip() in tag2idx[0].keys()) and (line.split(',')[i+1+num_task].strip() in tag2idx[0].keys())]))
    
    if params.group_classes:
        guess = []
        ans = []
        a = open(f, 'r').read().splitlines()
        for line in a:
            if len(line.split(',')) > 1:
                b = line.split(',')[1].strip() 
                c = line.split(',')[2].strip()
                if (str(b).strip() not in tag2idx[0].keys()) or (str(c).strip() not in tag2idx[0].keys()):
                    pass
                else:
                    ans.append(b)
                    guess.append(c)

        group_report = sklearn.metrics.classification_report(
            ans, guess, labels=["CD", "ST", "O"], output_dict=True)
        


    num_predicted, num_correct, num_gold = 0, 0, 0
    if num_task != 2:
        for i in range(num_task):
            num_predicted += len(y_pred[i][y_pred[i]>1])
            num_correct += (np.logical_and(y_true[i]==y_pred[i], y_true[i]>1)).astype(np.int).sum()
            num_gold += len(y_true[i][y_true[i]>1])





    elif num_task == 2:  
        num_predicted += len(y_pred[0][y_pred[0]>1])
        num_correct += (np.logical_and(y_true[0]==y_pred[0], y_true[0]>1)).astype(np.int).sum()
        num_gold += len(y_true[0][y_true[0]>1])
        
    print("num_predicted:{}".format(num_predicted))
    print("num_correct:{}".format(num_correct))
    print("num_gold:{}".format(num_gold))
    try:
        precision = num_correct / num_predicted
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.4f_R%.4f_F1%.4f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open(f, "r").read()
        fout.write("{}\n".format(result))

        fout.write("precision={:4f}\n".format(precision))
        fout.write("recall={:4f}\n".format(recall))
        fout.write("f1={:4f}\n".format(f1))

    os.remove(f)

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    if not params.group_classes:
        return precision, recall, f1, valid_loss
    else:
        return precision, recall, f1, valid_loss, group_report

if __name__ == "__main__":

    if params.wandb:
        wandb.init(project="news_bias", name=params.run)

    # config = BertConfig.from_pretrained('bert-base-cased')

    model_bert = BertMultiTaskLearning.from_pretrained('bert-base-uncased')
    print("Detect ", torch.cuda.device_count(), "GPUs!")
    # print("First Time cached is {}\n allocated is {}".format(
        # torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))
    model_bert = nn.DataParallel(model_bert)
    # print("cached is {}\n allocated is {}".format(torch.cuda.memory_cached(0),torch.cuda.memory_allocated(0)))
    
    # torch.cuda.empty_cache()

    # print("cached is {}\n allocated is {}".format(
    # torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))

    model_bert = model_bert.to(dev)

    # print("cached is {}\n allocated is {}".format(torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))


    train_dataset = PropDataset(params.trainset, False)
    eval_dataset = PropDataset(params.validset, True)
    # test_dataset = PropDataset(params.testset, True)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=params.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=params.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    # test_iter = data.DataLoader(dataset=test_dataset,
    #                             batch_size=params.batch_size,
    #                             shuffle=False,
    #                             num_workers=1,
    #                             collate_fn=pad)

    # print("cached is {}\n allocated is {}".format(
    #     torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))


    warmup_proportion = 0.1
    num_train_optimization_steps = int(
        len(train_dataset) / params.batch_size) * params.n_epochs
    param_optimizer = list(model_bert.named_parameters())
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

    optimizer = AdamW(optimizer_grouped_parameters, lr = params.lr, correct_bias=True )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(warmup_proportion*num_train_optimization_steps), num_training_steps = num_train_optimization_steps )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    binary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([3932/14263]).to(dev))

    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=params.patience, verbose=True)

    for epoch in range(1, params.n_epochs+1):
        # print("For epoch {} cached is {}\n allocated is {}".format(epoch, 
        #     torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))

        print("=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('results'):
            os.makedirs('results')
        fname = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run)
        spath = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run+".pt")

        # print("For epoch {} cached is {}\n allocated is {}".format(epoch, torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))

        train_loss = train(model_bert, train_iter, optimizer,
                           criterion, binary_criterion)

        
        avg_train_losses.append(train_loss.item())


        if not params.group_classes:
            precision, recall, f1, valid_loss = eval(
                model_bert, eval_iter, fname, criterion, binary_criterion)
            avg_valid_losses.append(valid_loss.item())


            if params.wandb:
                wandb.log({"Training Loss": train_loss.item(), "Validation Loss": valid_loss.item(
                ), "Precision": precision, "Recall": recall, "F1": f1})
        else:
            precision, recall, f1, valid_loss, group_report = eval(
                model_bert, eval_iter, fname, criterion, binary_criterion)
            avg_valid_losses.append(valid_loss.item())

            if params.wandb:
                wandb.log({"Training Loss": train_loss.item(), "Validation Loss": valid_loss.item(
                ), "Precision": precision, "Recall": recall, "F1": f1, "CD_F1": group_report["CD"]["f1-score"], "ST_F1": group_report["ST"]["f1-score"], "O_F1": group_report["O"]["f1-score"]})


        epoch_len = len(str(params.n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        early_stopping(-1*f1, model_bert, spath)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    res = os.path.join('results', timestr)
    # load the last checkpoint with the best model
    model_bert.load_state_dict(torch.load(spath))


    # precision, recall, f1, test_loss = eval(
    #     model_bert, test_iter, res, criterion, binary_criterion)
    # print_msg = (f'test_precision: {precision:.5f} ' +
    #              f'test_recall: {recall:.5f} ' +
    #              f'test_f1: {f1:.5f}')
    # print(print_msg)
