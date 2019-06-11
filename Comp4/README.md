CS565600 Deep Learning<br/>DataLab Cup 4: Reverse Image Caption
===
### Team50: Overfitting
### Members: 104062226 王科鈞 104062101 劉芸瑄	104000033 邱靖雅 104062315 李辰康

### Preprocessing
在 Data preprocessing 方面，我們依照論文所述之方式並再加以修改，而以下為我們對 image data 和 text data 做 preprocessing 的方法。
#### Image
在觀察過大多數的 train & test 圖片以後，我們發現圖片中花的中心位置大多位於圖片中央，因此我們將圖片中心部分的正方形裁切並保留下來，其餘地方捨棄掉，以減少我們圖片縮放後失真的比率。以下是我們所做的 preprocessing 步驟：

1.	Central cropping to a square with side length equals to the shorter side of the original image by **tf.image.resize_image_with_crop_or_pad**
2.	Scaling each sides to 76/64 times as big as the target image’s length of sides
3.	Random flipping horizontally by **tf.image.random_flip_left_right**
4.	Random cropping to target image size by **tf.random_crop**
5.	Normalizing each pixel to \[-1, 1\]

其中，我們在讀取 image 時的 function 是使用`cv2.imread`，而用此 fucntion 出來的圖片是用[ b,g,r ]來表示，所以我們將之轉換回 [ r,g,d ] 表示法。這樣來符合我們後續在 GAN 中產生顏色的方式。
<a id='Raw-Text'></a>
#### Raw Text
我們在觀察 word2Id_dict 及 id2Word_dict 時，發現助教的 word2Id_dict 中，`<pad>`對應到的 id 永遠是 5427，但`<pad>`的對應 id 其實有兩個 0 和 5427，而且 我們還發現 id 5428 並沒有任何 word 對應到，因此我們將 id 0 和 5428 從 word2Id_dict 中剔除，以此減少我們  rnn_encoder 學到完全不重要的 word。

其餘，我們還有將圖片和文字做重新配對，方法如下：
1. 下載原始的文字敘述來使用，由於原始檔案會根據圖片所屬的Class來擺置資料夾。
2. 接著，我們將所有*.txt丟到同個資料夾，再依助教提供的Dataframe中圖片順序串接成單一個train_captions.txt檔。
3. 再來，Testing Data的部分，我們必須先把助教提供的單字ID轉換為文字，但原始的文字已先經過nltk處理，只保留字根的，為了找到每個下載的.txt檔中哪個句子才是助教指定的，我們把每個句子轉換成保留單字字首的格式：例如：this flower has petals that are white with a small stigma→tfhptawwass，比較這個字串就能找出相同的句子。
4. 最後將所有Testing Data也串成一個test_captions.txt。

## Word Embedding
我們使用 [char-cnn-rnn](https://arxiv.org/pdf/1605.05395.pdf) 模型來實作word embedding。

實作方法：我們使用cnn先對原文的詞向量以df_dim進行covolution，來使詞向量變成經過卷積的抽象含義序列。
接著我們使用對經過卷積結果做lstm，其效果會優於傳統lstm，在這個過程中，cnn是用於提取特徵。

### CNN encoder
提取特徵，作為input做lstm。
```python
net_h0 = Conv2d(self.inputs, 4, 4, df_dim, 2, 2, name='cnnf/h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)

net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='cnnf/h1/conv2d', padding='SAME')
net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h1/batch_norm')

net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='cnnf/h2/conv2d', padding='SAME')
net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h2/batch_norm')

net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='cnnf/h3/conv2d', padding='SAME')
net_h3 = batch_normalization(net_h3, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h3/batch_norm')

net_h4 = flatten(net_h3, name='cnnf/h4/flatten')
net_h4 = fc(net_h4, num_out=self.t_dim, name='cnnf/h4/embed', biased=False)
```
### RNN
使用lstm cell
```python
LSTMCell = tf.contrib.rnn.BasicLSTMCell(self.t_dim, reuse=self.reuse)
initial_state = LSTMCell.zero_state(self.batch_size, dtype=tf.float32)
            
rnn_net = tf.nn.dynamic_rnn(cell=LSTMCell,inputs=embedded_word_ids,initial_state=initial_state,dtype=np.float32,time_major=False,scope='rnn/dynamic')
```


## Network Architecture
![](https://i.imgur.com/m0TnUxY.jpg)

### Algorithm
![](https://i.imgur.com/FldulLh.png)


### Generator
我們參考了[StackGan](https://github.com/hanzhanggit/StackGAN)使用其stage1的generator來實作generator的部分，![](https://i.imgur.com/fafOh5a.png)

Stage I Generator為圖中的State-I Generator G0 for sketch（藍色部分），將Augmented之後的結果和random的noise接起後經過fcn再經過upsampling，最後再經過tanh得到結果。

### Discriminator
#### Loss Function
在計算 discriminator & generator 的 loss , d_loss, g_loss，以 d_loss 為例，我們參考論文中一樣使用 `tf.nn.sigmoid_cross_entropy_with_logits` 來對 real_image, mismatch_image, fake_image 去分別跟同 shapes 的 one_likes, zero_likes, zero_likes 做 loss 計算。讓 real_image 的 logistic probability 更接近 1，然後讓 mismatch, fake image 的 logistic probability 更接近 0。而為何使用 sigmoid function + cross_entroy 的原因是，我們的產生目標是類似於 multiclass 的，因為要產生有變化性的 output images。最後我們將 d_loss 總合用 real_image d_loss + ( mismatch_image d_loss + fake_image d_loss) * 0.5。g_loss 同上用同一個 loss 計算的 function。


## Result
![](https://i.imgur.com/uqNeiCd.png)

1. the flower has a lot of petals mostly in orange color but the middle portion is more on yellow color.
2. the flower has many yellow petals surround the red stamen.
3. this flower has shiny red petals black filaments and a brown pistil.
4. this flower has a bright pink coloring with white around the edges and in the center.
5. flower has violet colored petals with purple anthers and pink filament.
6. the flower shown has pink and white petals with green pedicel.
7. this flower has petals that are overlapping and yellow with yellow stamen.
8. this flower is yellow in color with petals that are skinny and long.


### Experiments

| Vocab Size    | rnn epoch    | total epoch    | Private Score
| ----------------- | --------------- | --------------- | --------------- |
| 7000 | 80   | 600   | 0.12491      |
| 5427 | 120  | 600   | 0.12657      |
| 7000 | 80   | 480   | 0.12699      |
| 5427 | 80   | 400   | 0.12719      |
| 7000 | 80   | 500   | 0.12738      |
| 5427 | 80   | 310   | 0.12739      |
| 7000 | 50   | 100   | 0.12781      |

我們發現total epoch數越大，效果越好；而我們認為rnn epoch數則與vocab size相關。

