# Propaganda Analysis in News Articles

Online media articles often have source induced biases that sway user opinions and perspectives. There is no system in common knowledge with explainable decisions that identifies and removes these, often subjective, biases and can be used across sources. In this work we have made progress towards making an end to end framework for Fine Grained detection of propaganda in News Articles and then Rewriting them with a Neutral Point of view. 

## Getting Started

To run the code for training with BERT as backbone simply clone the repository and run 
```
python3 train.py --training --bert
```
Additional Parameters Used are:
Argument | Default Value |
---|---|
Batch Size | 16 |
Learning Rate | 3*10<sup>-5</sup>|
Group Classes | True |
Device | Cuda |

### Prerequisites

* `Pytorch <=1.40`
* `wandb`

### Different Experiments and Architectures
* [BERT Model 18 Class output(Both sentence and Token Level Classification)](https://github.com/chinmay-singh/Propaganda/blob/master)
* [BERT + CRF](https://github.com/chinmay-singh/Propaganda/blob/crf)
* [BERT Grouped into 3 Classes of Bias](https://github.com/chinmay-singh/Propaganda/tree/less)
* [BERT + Auxiliary objective of Valence Arousal and Dominance prediction](https://github.com/chinmay-singh/Propaganda/tree/lexicon)


## Running the tests

Set the Training Flag to False and change the input path to the Dev/Test Dataset


## Authors

* **Chinmay Singh** - [chinmay-singh](https://github.com/chinmay-singh)
* **Ayush Kaushal** - [ayushk4](https://github.com/ayushk4)

## Acknowledgments

* Vitobha Munigala, IBM Research
* Nishtha Madan, IBM Research

