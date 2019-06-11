Student ID, name of each team member.
- The models you tried during competition. 
- List the experiment you did. For example, how you define your reward, different training strategy, etc.
- Other things worth to mention. For example, Any tricks or techniques you used in this task.
- Conclusions (interesting findings, pitfalls, takeaway lessons, etc.)

CS565600 Deep Learning<br/>DataLab Cup 5: You draw I draw
===
### Team50: Overfitting
### Members: 
- 104062226 王科鈞 
- 104062101 劉芸瑄
- 104000033 邱靖雅
- 104062315 李辰康



## The Big Picture

在本次的 competiton 中，我們被要求訓練出一個進行 You draw, I draw 遊戲的 model。遊戲規則大致上為由兩個 player 輪流在一張圖上畫下比劃，並試著讓結果盡量接近自己的目標 class。在畫完後由一個 evaluation model 來評斷結果屬於哪一個 class。
本次競賽的 dataset 為四個目標 class 的 sketch，我們在使用這些 data 可以訓練出一個會畫四個 class 其中之一的 model。由於遊戲真正進行時，會有另外一個 model 輸出筆畫，因此單就會畫一個 class 的 model 沒有辦法在遊戲中取得好的表現。我們需要對原先只會畫一個 class 的 player 對不同的對手進行 fine tune，使之獲得對抗畫另一個 class 的 player 的能力。
我們本次競賽使用的方法是在 Play game 的過程中，收集由 evaluation model 判定的結果，作為 "Reward" 來 update 我們 SRNN 的 weight。下面的章節將會分別討論到我們架構的不同部分，可分為以下幾點：

1. SRNN
2. Evaluation Model
3. Play Game function

## SRNN

在這次的 competition 中我們使用 Sketch RNN 來進行筆畫的對戰。SRNN 的架構基本上與助教提供的相同，只有在 loss 上面有進行修改。
SRNN 的 output 其中有一項是下一筆畫的機率分佈，我們 fine tune 的目標就是讓「會使我們獲勝」的筆畫出現的機率越高越好。
觀察 SRNN 的 loss 可以發現它是由 pen_loss 以及 delta_xy_loss 組成的，其中 delta_xy_loss 是 output 筆畫的機率分布的 log 加上負號。由於訓練的目標是要最小化 model 的 loss，而 delta_xy_loss 越小，代表的就是該筆 output 的機率會越高（因為 loss 前面的負號）。以此在 play game 過程中，如果某一筆畫會增加 evaluation model 判定 player 勝利的話，我們就要將 loss 減少。實現的方法基本上都是將某一個 reward 以 tf.placeholder 的方式傳入 SRNN model 中，乘以原先的 loss，在每一輪遊戲後對 SRNN 進行一次 training。其中我們又試過調整 reward 的計算方式，有以下的幾種變形：

1. 若 evaluate 為我們獲勝則 reward = 1，否則為零
2. 將 evalaute model 判定四個 class 的 probability 拿出來，將 player 目標 class 的機率減對手 class 的機率

其中第二個方法背後的理論為這樣的 reward 比起第一種，可以刺激 player 的 model 不只是要畫「讓自己贏」的筆畫，還要畫「讓自己更容易贏，對手更不容易贏」的筆畫。實驗結果也確實有些微的進步。

## CNN Classifier

我們訓練一個CNN模型作為 Evaluate model，在 play game 的過程中使用這個 Evaluate model 作為判斷標準，對畫畫對戰的結果圖進行 Classification。為了有效利用此 CNN 來增進我們的 Agent 在畫某一特定物品 (ex:Balloon, Bulb) 時能畫得更好，我們對於 CNN 的 Training data 做了一些 Preprocessing。

由於在 Game 遊玩過程中，會有 Evaluation 這一項操作，也就是在玩家雙方都各畫 4 筆之後，我們會對之後每筆當下的結果圖做 Classification ，以給予雙方玩家不同的 Rewards or Punishments。也因為上述規則，我們必須讓我們的 Evaluation model 能分辨出這些結果圖，但我們原先的 Training data 皆是遊玩最後的結果圖，因此我們必須另外去產生這些半途過程中的結果圖。

###  Generate Extra Training Data
我們讓我們將原先 Training Data 的每張圖的全部筆畫，去依照從第4筆開始，每增加一筆就額外將他們切割出然後以 .npy 檔存下來，所以最後我們就會有 [(1-4),(1-5),(1-6),...,(1-總筆劃)]，這樣形式的每個類別的 Training Data。

### Train Our Model On The Data
我們使用原始 Evaluation model 的 CNN 架構，但調整 keep rate 至 0.75，然後因為原  Model 的 Input 是吃 image 的，所以我們必須將處裡過的筆畫切割 Data 做 Renderring，以此讓我們能餵進我們想要的結果圖到 CNN 中去做 Training。

```python=
    class Evaluate_model:
        def __init__(self, image_size, model_name='evaluate_model', recover=None, keep_rate=0.7, train=True):
            self.train = train
            self.ckpt_path = './checkpoints/'
            self.model_name = model_name
            self.keep_rate = keep_rate

            with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
                self.input_x = tf.placeholder(
                  tf.float32, [None, image_size, image_size], name="input_x")

                self.labels = tf.placeholder(tf.int32, [None], name="input_y")


                hidden = tf.reshape(self.input_x, [-1, image_size, image_size, 1])
                hidden = tf.layers.conv2d(
                  hidden,
                  filters=64,
                  kernel_size=5,
                  activation=tf.nn.relu,
                  padding='same')
                hidden = tf.nn.lrn(
                  hidden, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

                hidden = tf.layers.conv2d(
                  hidden,
                  filters=64,
                  kernel_size=5,
                  activation=tf.nn.relu,
                  padding='same')
                hidden = tf.nn.lrn(
                  hidden, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

                hidden = tf.layers.max_pooling2d(
                  hidden, pool_size=3, strides=2, padding='same')
                hidden = tf.layers.conv2d(
                  hidden,
                  filters=128,
                  kernel_size=5,
                  activation=tf.nn.relu,
                  padding='same')
                hidden = tf.nn.lrn(
                  hidden, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

                hidden = tf.layers.conv2d(
                  hidden,
                  filters=128,
                  kernel_size=5,
                  activation=tf.nn.relu,
                  padding='same')
                hidden = tf.nn.lrn(
                  hidden, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')

                hidden = tf.layers.max_pooling2d(
                  hidden, pool_size=3, strides=2, padding='same')

                hidden = tf.layers.flatten(hidden)

                hidden = tf.layers.dense(hidden, 384)
                hidden = tf.layers.dense(hidden, 192)
                hidden = tf.layers.dense(hidden, 96)

                # Add dropout
                with tf.variable_scope("dropout"):
                    hidden = tf.layers.dropout(hidden, 1 - self.keep_rate)

                hidden = tf.layers.dense(hidden, 4)

                # label of balloon: 0
                # label of bulb: 1
                # label of ice: 2
                # label of microphone: 3
                self.predictions = tf.nn.softmax(hidden)

                self.accuracy = tf.reduce_mean(
                  tf.to_float(tf.equal(tf.cast(tf.argmax(self.predictions, 1), tf.int32), tf.cast(self.labels, tf.int32))))

                label_one_hot = tf.one_hot(tf.cast(self.labels, tf.int32), 4, dtype=tf.float32)

                # CalculateMean cross-entropy loss
                losses = tf.nn.softmax_cross_entropy_with_logits(
                  logits=hidden, labels=label_one_hot)
                self.loss = tf.reduce_mean(losses)

                for v in tf.trainable_variables():
                    self.loss += 0.001 * tf.nn.l2_loss(v)

                # Define Training procedure
                self.lr = tf.Variable(0.001, trainable=False)
                optimizer = tf.train.AdamOptimizer(self.lr)
                grads_and_vars = optimizer.compute_gradients(
                  self.loss,
                  var_list=[v for v in tf.global_variables() if model_name in v.name])
                self.train_op = optimizer.apply_gradients(grads_and_vars)

            self._get_session()
            # self._get_train_data_iter()
            self._init_vars()
            self._get_saver()

            if recover is not None:
                self._load_checkpoint(recover)


        def _get_session(self):
            self.sess = tf.Session()

        def _init_vars(self):
            self.sess.run(tf.global_variables_initializer())

        def _get_saver(self):
            self.saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if self.model_name in v.name])
            # self.saver = tf.train.Saver()

        def _load_checkpoint(self, recover):
            if self.train:
                self.saver.restore(self.sess, self.ckpt_path + 'classifier_model_' + str(recover) + '.ckpt')
            else:
                self.saver.restore(self.sess, self.ckpt_path + 'classifier_model_' + str(recover) + '.ckpt')
            print('-----success restored checkpoint--------')

        def _save_checkpoint(self, epoch):
            self.saver.save(self.sess, self.ckpt_path + 'classifier_model_' + str(epoch) + '.ckpt')
            print('-----success saved checkpoint--------')

        def _get_train_data_iter(self):
            if self.train: # training data iteratot
                iterator_train, types, shapes = data_iterator('./train_images.pkl',
                                                          BATCH_SIZE, training_data_generator)
                iter_initializer = iterator_train.initializer
                self.next_element = iterator_train.get_next()
                self.sess.run(iterator_train.initializer)
                self.iterator_train = iterator_train

            else: # testing data iterator
                iterator_test, types, shapes = data_iterator('./test_images.pkl', BATCH_SIZE, training_data_generator)
                iter_initializer = iterator_test.initializer
                self.next_element = iterator_test.get_next()
                self.sess.run(iterator_test.initializer)
                self.iterator_test = iterator_test

        def training(self):
            N_EPOCH = 10

            # read dataset
            df = pd.read_pickle('train_sketches.pkl')
            sketches = df['sketches'].values
            labels = df['labels'].values

            N_SAMPLE = len(labels)
            N_BATCH_EPOCH = int(N_SAMPLE/BATCH_SIZE) - 1

            for _epoch in range(N_EPOCH):
                start_time = time.time()

                for _step in range(N_BATCH_EPOCH):
                    step_time = time.time()

                    start_idx = _step * BATCH_SIZE
                    end_idx = start_idx + BATCH_SIZE

                    sketch_batch = sketches[start_idx:end_idx]
                    label_batch = labels[start_idx:end_idx]

                    # render images in batch
                    # 3 to 5
                    # preprocess sketches
                    sketch_batch = preprocess(sketch_batch)

                    # render
                    # calculate normailizing factor
                    normalizing_scale_factor = calculate_normalizing_scale_factor(sketch_batch)

                    # normalize dataset
                    train_sketches = normalize(sketch_batch, normalizing_scale_factor)

                    # convert to stroke-5 format
                    sketch_batch = to_big_sketches(sketch_batch)

                    image_batch = []
                    for sketch_entry in sketch_batch:
                        seq_len = len(sketch_entry)
                        image_batch.append(render_imgs(np.array([sketch_entry]), IMAGE_SIZE, seq_len, 0)[0])


                    _, self.error = self.sess.run([self.train_op, self.loss], feed_dict={
                                                                    self.input_x: image_batch,
                                                                    self.labels: label_batch})


                    if _step % 50 == 0:
                        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, loss: " \
                                % (_epoch, N_EPOCH, _step, N_BATCH_EPOCH, time.time() - step_time),
                                   self.error)


                if _epoch != 0:
                    self._save_checkpoint(_epoch)

        def testing(self):

            # read dataset
            df = pd.read_pickle('test_sketches.pkl')
            sketches = df['sketches'].values
            labels = df['labels'].values

            N_SAMPLE = len(labels)
            N_BATCH_EPOCH = int(N_SAMPLE/BATCH_SIZE) - 1

            acc_list = []

            for _step in range(N_BATCH_EPOCH):

                start_idx = _step * BATCH_SIZE
                end_idx = start_idx + BATCH_SIZE

                sketch_batch = sketches[start_idx:end_idx]
                label_batch = labels[start_idx:end_idx]

                # render images in batch
                # 3 to 5
                # preprocess sketches
                sketch_batch = preprocess(sketch_batch)

                # render
                # calculate normailizing factor
                normalizing_scale_factor = calculate_normalizing_scale_factor(sketch_batch)

                # normalize dataset
                train_sketches = normalize(sketch_batch, normalizing_scale_factor)

                # convert to stroke-5 format
                sketch_batch = to_big_sketches(sketch_batch)

                image_batch = []
                for sketch_entry in sketch_batch:
                    seq_len = len(sketch_entry)
                    image_batch.append(render_imgs(np.array([sketch_entry]), IMAGE_SIZE, seq_len, 0)[0])

                test_accuracy = self.sess.run(self.accuracy, feed_dict={
                                                                    self.input_x: image_batch,
                                                                    self.labels: label_batch})
                acc_list.append(test_accuracy)

            print("Testing Accuracy : %.3f" %np.mean(acc_list))       
```

### Test And Result
我們利用另外利用原 Testing data 去做一樣的 Data 拓展，然後去測試我們的 Evaluation 的 Accuracy，最後測得的結果是 0.82。


## Play Game Function

### 前處理
在原本助教提供的play game function中，我們必須修改部分來加入SRNN更新reward的部分。
原本input_sequence為之前畫的每一筆與這次所產生的比畫做concstenate產生input_sequence。
```python
input_sequence = np.concatenate([input_sequence, generated_stroke], axis=0)
```
而為了能夠作為srnn的input sequence( 其shape為[None, hps.max_seq_len + 1, 5]分別是batch size, max length of input sequence及5(dx, dy 及三個state))在資料上需要做一些前處理。

```python
input_sequence_srnn = pad_data(input_sequence_srnn.reshape(1, input_sequence_srnn.shape[0], input_sequence_srnn.shape[1]), hps.max_seq_len)
```

#### pad data
```python
def pad_data(sketches, max_seq_len):
  """Pad the batch to be stroke-5 bigger format as described in paper."""
  result = np.zeros((len(sketches), max_seq_len + 1, 5), dtype=float)
  for i in range(len(sketches)):
    l = len(sketches[i])
    result[i, 0:l, :] = sketches[i][:, :]
    result[i, l:, 4] = 1

    # put in the first token, as described in sketch-rnn methodology
    result[i, 1:, :] = result[i, :-1, :]
    result[i, 0, :] = 0
    result[i, 0, 2] = 1  # setting S_0 from paper.
  return result
```
先將data reshape成(1, stroke num, 5)再把它pad到max_seq_len，以符合原本training的shape做接下來srnn的training。

### SRNN Training
```python
# if it is player 1's turn
if turn % 2 == 0 and not player1_stop:
    # if the probability of classifier between the two class is higher than a threshold
    if predictions[0, class_to_label[class_to_draw1]]  - predictions[0, class_to_label[class_to_draw2]] > 0.3:
        reward = predictions[0, class_to_label[class_to_draw1]]
        sess.run(player1[dict_key1].train_op, feed_dict={
                                        player1[dict_key1].input_sequence: input_sequence_srnn,
                                        player1[dict_key1].batch_size: np.asarray(1),
                                        player1[dict_key1].reward: np.asarray(reward)})
```
如前面SRNN所提，我們將evaluate model(classifier)的機率結果``` predictions[0, class_to_label[class_to_draw1]]```作為reward來更新我們player 1的SRNN model。
將處理過後的input dict、batch size和我們最後得到reward機率作為input feed dict到SRNN model中做training。
```python
sess.run(player1[dict_key1].train_op, feed_dict={
        player1[dict_key1].input_sequence: input_sequence_srnn,
        player1[dict_key1].batch_size: np.asarray(1),
        player1[dict_key1].reward: np.asarray(reward)})
```

