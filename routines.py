import torch
from params import params
import numpy as np
from sklearn.metrics import average_precision_score
import sklearn


def train(model, iterator, optimizer, scheduler):
    
    
    model.train()
    
    train_losses = []

    for k, batch in enumerate(iterator):

        words, x, is_heads, att_mask, tags, y, seqlens = batch
        

        optimizer.zero_grad()

        loss = model(x, attention_mask=att_mask, labels = y)

        loss.backward()

        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if k%100 == 0:
            print("step: {}, loss: {}".format(k,loss[0].item()))

    train_loss = np.average(train_losses)

    return train_loss


def eval(model, iterator):

    model.eval()

    valid_losses = []

    Words, Is_heads = [ ], [ ]
    Tags = [ ]
    Y = [ ]
    Y_hats = [ ]


    with torch.no_grad():
        for _, batch in enumerate(iterator):
            x,  y, att_mask = batch

            outputs = model(x, attention_mask=att_mask, labels=y)

            loss = outputs[0]

            logits = outputs[1]

            y_hats = logits.argmax(-1)

            valid_losses.append(loss.item())

            Y.extend(y.cpu().numpy().tolist())
            Y_hats.extend(y_hats.cpu().numpy().tolist())

        
    valid_loss = np.average(valid_losses)


    group_report = sklearn.metrics.classification_report(
                    Y, Y_hats, output_dict=True)

    precision   =   group_report['weighted avg']['precision']

    recall = group_report['weighted avg']['recall']
    
    f1 = group_report['weighted avg']['f1-score']
    
    

    print("precision=%.4f" % precision)
    print("recall=%.4f" % recall)
    print("f1=%.4f" % f1)
    
    return precision, recall, f1, valid_loss, group_report
