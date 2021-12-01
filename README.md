# MCNN: Inferring RNA-protein binding sites using multiple convolutional neural networks
Computational algorithms for identifying RNAs that bind to specific RBPs are urgently needed, and they can complement high-cost experimental  methods.  In this study, we develop a convolutional neural network (CNN) based method called MCNN  to predict RBP binding sites using mutiple RNA sequences processed by different windows size. 

# Dependency:
python 3.8.5
Pytorch 1.4.0

# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and decompress it in current dir. It has 24 experiments of 21 RBPs, and we need train one model per experiment.

# Supported GPUs
Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs. In addition, MCNN can also be adapted to protein binding sites on DNAs and identify DNA binding speciticity of proteins. 

# Usage:
python ideepe.py [-h] [--posi <postive_sequecne_file>]
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE]
                 [--out_file OUT_FILE] [--train TRAIN] [--model_file MODEL_FILE]
                 [--predict PREDICT] [--testfile TESTFILE] [--batch_size BATCH_SIZE] <br>
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS] <br>
It supports model training, testing.

# Use case:
Take ALKBH5 as an example, if you want to predict the binding sites for RBP ALKBH5.
You first need train the model for RBP ALKBH5, then the trained model is used to predict binding probability of this RBP for your sequences. The follwoing CLI will train a  model using mutiple CNNs, which are trained using positves and negatives derived from CLIP-seq.

# step 1:
1. python mcnn.py --posi=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa  --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CNN --train=True --n_epochs=50 
MCNN saves multiple models according length difference X of window 
For example, when X=200, it will save "model.pkl.101", "model.pkl.301" and "model.pkl.501". The smaller the X, the greater the number of saved models. 

# step 2:
2. python mcnn.py --testfile=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa --model_type=CNN 
--predict=True

testfile is your input fasta sequences file, and the predicted outputs for all sequences will be defaulted saved in "prediction.txt". The AUC outputs for all sequences will be saved in "pre_auc.txt". The training time and testing time for all sequences will be saved in "time_train.txt" and "time_test".  The value in each line corresponds to the probability of being RBP binding site for the sequence in fasta file. NOTE:if you have positive and negative sequecnes, please put them in the same sequecne file, which is fed into model for prediciton. DO NOT predict probability for positive and negative sequence seperately in two fasta files, then combine the prediction.

# NOTE
When you train iDeepE on your own constructed benchmark dataset, if the training loss cannot converge, may other optimization methods, like SGD or RMSprop can be used to replace Adam in the code. 

# Contact
Zhengsen Pan: zhengsenpan@foxmail.com

# Updates:
1/12/2021
