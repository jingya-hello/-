# Contest 2 : Object Detection
## Student ID, name of each team member
- 104062101 劉芸瑄
- 104000033 邱靖雅
- 104062226 王科鈞
- 104062315 李辰康

# Private score: 3th

# Overview : 
In this competition, we use a model called DSOD to complete the object detection task.

## Deeply Supervised Object Detector (DSOD)

It is a framework that can learn object detectors from scratch.

#### Question : Did you pre-train your model?
NO, we don't. The model we use is DSOD and it features on proposal-free and there's no ROI pooling.
Therefore, there's no need to pretrain model since we don't need to initialize the layer before ROI pooling. Please refer to the details in the design principle1.

### Design Principle
There are four design priciple of DSOD
#### Principle 1 : Proposal-Free

ROI pooling generates features for each region proposals that hinder the gradients, therefore methods with ROI pooling usually needs work with pre-trained models which initialize the layers before ROI pooling. Therefore, the author proposed a Proposal-free model which can converge successfully without any pre-trained models.

#### Principle 2 : Deep Supervision

DSOD use a block called dense block that use dense layer-wise connection(as in DenseNet) which all preceding layers in the block are connected to the currect layer.
The pros of dense block is that all layers can share the supervised signals.

#### Principle 3 : Stem Block

the paper define stem block which includes three 3x3 convolution layers (first stride = 2, others stride = 1) and follows a 2x2 max pooling layer.
The pros of stem block can reduce the information loss from raw input images.

#### Principle 4 : Dense Prediction Structure

Six scales of feature maps are applied for predicting objects.
The Scale-1 feature maps are from middle layer(38x38) are used to handle the small objects.
The other 5 layers are on top of the backbone sub-network.
Then, a transition layer is adopted between two contiguous scales of feature maps.

##### Learning Half and Reusing Half
In each scale, half of the feature maps are learned from the previous scale with a series of conv-layers, and the other half are down-sampled from the contiguous high-rersolution feature maps.

## Implementation

#### Question : How did you build your model(What model, any techniques you used, what obstacles you met and how did you deal with them)?

We use DSOD model and the technique is describe in the design principle and implementation session.

The architecture is as below:
![](https://i.imgur.com/iUATMOG.png)

The visualize architecture is as below:
![](https://i.imgur.com/LRbnts9.png)


### Stem block : 
```python
 # the first conv layer of the stem block which is stride 2 as the architecture said
model = L.Convolution(net[from_layer], kernel_size=3, stride=2, num_output=64, 
                          pad=1, bias_term=False, weight_filler=dict(type='xavier'),
                          bias_filler=dict(type='constant'))

model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)])
model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
model = L.ReLU(model, in_place=True)
 ```
The second and third layer are same as the first layer but the stride is 1.
After a stack of three 3x3 conv, it is followed by a max pooling layer
 ```python
# do a max pooling
model = L.Pooling(model, pool=P.Pooling.MAX, kernel_size=2, stride=2)
 ```
 
 ### Dense block
 We first construct a function bn_relu_conv as below : it do a batch norm , then use ReLU as activation function and finally do convolution:
 ```python
 def bn_relu_conv:
    batch_norm = L.BatchNorm(bottom, in_place=False,
                            param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                            dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    relu = L.ReLU(scale, in_place=inplace)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                            num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant'))
 ```
 The first dense block:
 ![](https://i.imgur.com/UFsDxhB.png)
 we use for loop and add_bl_layer to add 6 * new bn_relu_conv
 ```python
 for i in range(6):
    model = add_bl_layer(model, growth_rate, dropout, 4)
    nchannels += growth_rate
 ```
 ### Transition Layer
 ![](https://i.imgur.com/jmvvhU8.png)
Do a 1*1 conv then do a max pooling with stride 2
```python
 def transition(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout, inplace=False)
    pooling = L.Pooling(conv, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    return pooling
```
### Other Dense Block and Transition Layer
The Transition w/o PoolingLayer is describe as below : we only do a batchnorm relu convolution and without doing max pooling 
```python
def transition_w_o_pooling(bottom, num_filter, dropout):
    conv = bn_relu_conv(bottom, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout, inplace=False)
    return conv
```
other layer are same as the dscription above:
![](https://i.imgur.com/jZax6Qc.png)
```python
# Dense Block2
for i in range(8):
    model = add_bl_layer(model, growth_rate, dropout, 4)
    nchannels += growth_rate
    # transition layer2
    nchannels = int(nchannels / times)
    model = transition_w_o_pooling(model, nchannels, dropout)  # 38x38
    net.First = model
    model1 = L.Pooling(model, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # Dense Block3
    for i in range(8):
        model1 = add_bl_layer(model1, growth_rate, dropout, 4)
        nchannels += growth_rate
    # transition w/o pooling layer1
    nchannels = int(nchannels / times)
    model1 = transition_w_o_pooling(model1, nchannels, dropout)
    # Dense Block4
    for i in range(8):
        model1 = add_bl_layer(model1, growth_rate, dropout, 4)
        nchannels += growth_rate
    # transition w/o pooling layer2
    model1 = transition_w_o_pooling(model1, 256, dropout)
    # 
```
### Prediction Layer
As the picture shows and as the fourth design principle, we know that half of the feature maps are learned from the previous scale with a series of conv-layers, and the other half are down-sampled from the contiguous high-rersolution feature maps.
So, we construct a function add_bl_layer2 that combine the feature map and the previous half feature map.
```python
def add_bl_layer2(bottom, num_filter, dropout, width):
    conv = bn_relu_conv(bottom, ks=1, nout=int(width*num_filter), stride=1, pad=0, dropout=dropout)
    conv = bn_relu_conv(conv, ks=3, nout=num_filter, stride=2, pad=1, dropout=dropout)
    conv2 = L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv2 = bn_relu_conv(conv2, ks=1, nout=num_filter, stride=1, pad=0, dropout=dropout)
    concate = L.Concat(conv2, conv, axis=1)
    return concate
```
The architecture:
![](https://i.imgur.com/KKRr40D.png)
```python
model2 = add_bl_layer2(model1, 256, dropout, 1) # pooling4: 10x10
net.Third = model2
model3 = add_bl_layer2(model2, 128, dropout, 1) # pooling5: 5x5
net.Fourth = model3
model4 = add_bl_layer2(model3, 128, dropout, 1) # pooling6: 3x3
net.Fifth = model4
model5 = add_bl_layer2(model4, 128, dropout, 1) # pooling7: 2x2
net.Sixth = model5
return net
```
### Obstacle
what obstacles you met and how did you deal with them?

Hard to build our model:
The architecture didn't show the detail architecture in the DSOD Prediction Layers, and we find out another picture describes the detail and finally implement it as the picture shows.

COCO and VOC2007 datasets have not the same corresponding classses:
Due to COCO has 80 classes of images and VOC2007 only hass 20 classes, so we must match the classes to one kind of data form. So our solution is to change the representation of data from COCO's to VOC2007's. Therefore we can easily build up the training data and testing data from the same data representing dataset. Also, this data pre-processing can make us generate a lmdb file in a simple way.



## Training Data

We use the training data below:
* Pascal VOC2007
* COCO 

Things we need to do is that COCO has 80 classes and VOC2007 has 20 classes : 
1. convert VOC to COCO type
2. match the output 

### Training time
36 hours

      
## Others
 #### Question : Anything you think it's worth mentioning(your findings, other works).
 
 We first tried the yolo model which we downloaded from github and tried to train it. We then train it for about 18 hours,  however the result isn't good.
 We think that the reason is that :
 1. yolo can only predict a best class for each grid when the threshold is high. However, if the threshold become too low, there will be too much noise in the prediction.
 2. In this competition the training data is not really big compared to the original yolo training dataset.
 
However, some students said that yolo can still produce good result, so maybe we can try yolo again.

## Dev Environment
Developed on PC with:
* OS: Ubuntu16.04
* GPU: TITAN X
* CUDA 7.5
