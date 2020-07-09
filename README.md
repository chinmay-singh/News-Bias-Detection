# Fine Grained Propaganda Detection


1. pip install -r requirements.txt 
2. For the training, Run one of the following: (Note that --lexi_loss_wt --senti_loss_wt is for scaling in the multi task loss. --alpha is b/w Fragment level loss and sentence level loss. Leading to upto 4 different losses.)

```python train.py --bert --training --lexi_loss_wt 10 --senti_loss_wt 3 --batch_size 16 --lr 3e-5 --n_epochs 20 --patience 7```

```python train.py --joint --training --lexi_loss_wt 10 --senti_loss_wt 3--batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```

```python train.py --granu --training --lexi_loss_wt 10 --senti_loss_wt 3 --batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```

```python train.py --mgn --sig --training --lexi_loss_wt 10 --senti_loss_wt 3--batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```

3. For the fragment-level evaluation, Run the evaluate.sh 

```./evaluate.sh ./result/[output.file] bert```

```./evaluate.sh ./result/[output.file] bert-joint```

```./evaluate.sh ./result/[output.file] bert-granu```

```./evaluate.sh ./result/[output.file] mgn```

<!-- 4. For the span-level evaluation, Run the span-evaluate.sh  ```./span-evaluate.sh ./result/[output.file] bert``` ```./span-evaluate.sh ./result/[output.file] bert-joint``` ```./span-evaluate.sh ./result/[output.file] bert-granu``` ```./span-evaluate.sh ./result/[output.file] mgn``` -->

## Versions:
Python 3.6.8, CUDA 10.1, Torch 1.0, huggingface/pytorch-pretrained-BERT 0.4


