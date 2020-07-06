import argparse

parser = argparse.ArgumentParser()

path_data = '.'
train_path = "/data/protechn_corpus_eval/train"
text_path = "/*.tsv"
label_path = "/*.txt"
dev_path = "/data/protechn_corpus_eval/dev"

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--multitask_feat_dims", type=int, default=512)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=7)
parser.add_argument("--training", dest="training", action="store_true")
parser.add_argument("--checkdir", type=str, default="checkpoints")
parser.add_argument("--resultdir", type=str, default="results")
parser.add_argument("--bert", dest="bert", action="store_true" , default=True)
parser.add_argument("--joint", dest="joint", action="store_true")
parser.add_argument("--granu", dest="granu", action="store_true")
parser.add_argument("--mgn", dest="mgn", action="store_true")
parser.add_argument("--sig", dest="sig", action="store_true")
parser.add_argument("--rel", dest="rel", action="store_true", default=True)
parser.add_argument("--trainset", type=str, default=path_data+train_path)
parser.add_argument("--validset", type=str, default=path_data+dev_path)
parser.add_argument("--input", type=str, default= None)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--group_classes", type=bool, default=False, help="Whether or not to group classes in task one")
parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
parser.add_argument("--wandb",  dest="wandb", action="store_true", default= False)

parser.add_argument("--sentence_level", type=bool, default=False, help="Calculate loss on sentence level, or on token level.")

# Losses
parser.add_argument("--lexical_loss_wt", type=int, default=0, help="Weight for lexical loss")
parser.add_argument("--lexical_ignore_range", type=float, default=0.01, help="Only ground truth values (> 0.5 + range) and ( < 0.5 - range) be considered for loss.")
parser.add_argument("--sentiment_loss_wt", type=int, default=0, help="Weight for sentiment loss")

params = parser.parse_args()
