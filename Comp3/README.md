# Contest 3 DL_comp3_50_report.ipynb

# Student ID, name of each team member
- 104062101 劉芸瑄
- 104062315 李辰康
- 104000033 邱靖雅
- 104062226 王科鈞

# Your code and some descriptions


# How did you preprocess your data (if you did it by yourself, not using the method on course notebook)
Nothing

# Do you use pre-trained model? (word embedding model, CNN model, etc.) 
我們在 word embedding 上嘗試使用了兩種 pretrained model：GoogleNews-vectors-negative300 和  gensim word2vec。

GoogleNews-vectors-negative300 的效果非常差，我們推測原因用來訓練該 model 的 document 與我們這次問題的相去甚遠，導致該 embedding 在我們的 model 中表現不好。

gensim word2vec model 則是仿造 Competition 1 的做法，先蒐集所有 training data 的 caption，把他們分解成 lists of words 在餵入 model 訓練。結果比我們最後採用的方法稍差。
** W2V.ipynb

# How do you design your image captioning model? (illustrate every step you design and why this works)

### (Network Architecture)
<a id='RNN-Cell'></a>

我們本次使用的 RNN 架構是 Tensorflow 的 BasicLSTMCell。在 build model 的時候 initial weight 為 0。在 cell 的外面在加上 DropoutWrapper，它會在每個Cell的Input、Output、H State Output加上Dropout。

在 training 階段 LSTM 的 input 除了 image features 外還有 input_seq 做 word2vec 的 sequence embedding。

### (Attention)



# RNN Tricks and Techniques We Used (or tried)

## Beam Search


## Weight Clipping

由於我們 output 的句子最大長度相當長，如果將 RNN 展開來看的話，gradient 其實需要 back prop 過很長的一段距離。這樣很容易因為 gradient 的一點波動產生 Exploding/Vanishing gradient 的問題。為了解決 exploding gradient 的問題，我們在 model 裡加入了 weight clipping。
Tensorflow 有相當多的 clipping 方法，clip_by_value 只是很直接的江過大或過小的值減掉，但這樣一來被 clip 掉的值便會失去原本與其他值的比例。我們最後選了 clip_by_norm 限制 L2-norm 來實作 weight clipping。

## Curriculum Learning

Curriculum learning 的原理就是讓 model 從簡單的 case 開始學起，漸漸增加難度，理論上便可以先很難 train 的問題比較快 converge。在我們的這個問題中，我們將「簡單」的問題定義為比較短的句子。我們透過 Tensorflow 的 Dynamic_RNN 裡的參數限制 RNN 能「看」 的字數。從 2 開始，每個 epoch 往上加。

就結果來看，使用 curriculum learning 確實能夠使我們的 loss 快速的下降。在加入 curriculum learning 之前，我們大概需要 train 50 個 epoch 才能讓 model 有較好的表現。但是在加入之後，只要約 20 個 epoch 便可以有相當好的成果。再者，由於在 max_length 較短的時候，RNN 跑的次數也低，因此每個 epoch 花的時間也比較短，training 的速度能進一步加快。

然而，加上這個機制也並不是沒有缺點，雖然 curriculum learning 能讓 model 比較快 converge，但是最後訓練出來的 model 的 performance 分數卻會比較低。當我們把他們 inference 的結果拿出來做比較，會發現經過 curriculum learning 的 model output 出來的句子偏短。這個當然有可能是我們 training 時間還不夠，導致 model 還沒有充分學到輸出長句的原因。

以下幾張圖為我們挑選出來，由我們最佳的 model 以及做 curriculum learning model 的輸出的比較，可以發現後者的答案雖然方向大致正確，但是句子的長度以及描述的豐富度還是較差的。

<br/>

![](https://i.imgur.com/c5BjsOy.jpg)

Our final model:
> a group of people playing a game with nintendo wii controllers

Model with Curriculum learning:
> a group of people playing a game of wii

<br/>

![](https://i.imgur.com/hkm4VsL.jpg)

Our final model:
> a man riding a snowboard down the side of a snow covered slope

Model with Curriculum learning:
> a man on a snowboard in the snow

## Attention


# Demo: take arbitrary one image and use your model to generate caption for it in your report.




# Conclusions (interesting findings, pitfalls, takeaway lessons, etc

