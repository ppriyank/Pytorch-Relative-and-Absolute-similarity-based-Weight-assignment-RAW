# Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW
Pytorch Implementation of the Paper A UNIFIED VIEW OF DEEP METRIC LEARNING VIA GRADIENT ANALYSIS
 
A UNIFIED VIEW OF DEEP METRIC LEARNING VIA GRADIENT ANALYSIS
*Anonymous authors  
Paper under double-blind review*

https://openreview.net/pdf?id=Skf5qiC5KQ

Some Terms explained in the paper:

* Hardest positive: The postive label example which has the max cosine similarity 
* Hardest negative: The negative example which has the min cosine similarity 
* Valid Positive Examples : The postive label examples which have smaller cosine distance compared to Hardest negative : P
* Valid Negative Examples : The negative label examples which have smaller cosine distance compared to Hardest positive : N



## Intution

![Visualization](https://github.com/ppriyank/Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW/blob/master/Screen%20Shot%202019-10-21%20at%203.11.28%20AM.png)

