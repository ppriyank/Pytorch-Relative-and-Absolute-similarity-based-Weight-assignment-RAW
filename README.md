# Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW
Pytorch Implementation of the Paper A UNIFIED VIEW OF DEEP METRIC LEARNING VIA GRADIENT ANALYSIS
 
A UNIFIED VIEW OF DEEP METRIC LEARNING VIA GRADIENT ANALYSIS
*Anonymous authors  
Paper under double-blind review*

https://openreview.net/pdf?id=Skf5qiC5KQ


## Intution
* Hardest positive: The postive label example which has the max cosine similarity 
* Hardest negative: The negative example which has the min cosine similarity 
* Valid Positive Examples : The postive label examples which have smaller cosine distance compared to Hardest negative : P
* Valid Negative Examples : The negative label examples which have smaller cosine distance compared to Hardest positive : N

   <img src="https://github.com/ppriyank/Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW/blob/master/Screen%20Shot%202019-10-21%20at%203.11.28%20AM.png" width="400">


## Weight : 

![Weights](https://github.com/ppriyank/Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW/blob/master/Screen%20Shot%202019-10-21%20at%203.16.17%20AM.png)

### where :
### * Relative similarity between the positive pair:
   <img src="https://github.com/ppriyank/Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW/blob/master/Screen%20Shot%202019-10-21%20at%203.24.26%20AM.png" width="300">

### * Absolute similarity
   <img src="https://github.com/ppriyank/Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW/blob/master/Screen%20Shot%202019-10-21%20at%203.27.02%20AM.png" width="200">
    
    
## Gradient Equivalent Loss function 

![loss](https://github.com/ppriyank/Pytorch-Relative-and-Absolute-similarity-based-Weight-assignment-RAW/blob/master/Screen%20Shot%202019-10-21%20at%203.23.03%20AM.png)





