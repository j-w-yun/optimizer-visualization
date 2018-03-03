# optimizer-visualization

## Visualize gradient descent optimization algorithms in Tensorflow.

For an overview of each gradient descent optimization algorithm, [visit this helpful resource](http://ruder.io/optimizing-gradient-descent/).

All methods start at the same location, specified by two variables. Both x and y variables are improved by the following Optimizers:

[Adadelta documentation](https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer)

[Adagrad documentation](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)

[Adam documentation](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

[Ftrl documentation](https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer)

[GD documentation](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)

[Momentum documentation](https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer)

[RMSProp documentation](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)

For an overview of each gradient descent optimization algorithms, visit [this helpful resource](http://ruder.io/optimizing-gradient-descent/).

#### Numbers in figure legend indicate learning rate, specific to each Optimizer.
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/movie5.gif)

#### A sharp gaussian depression at x=0, y=0. Note the optimizers' behavior when gradient is steep.
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/movie6.gif)

## Additional Figures

![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/movie3.gif)

![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/movie2.gif)

![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/movie.gif)

#### AdadeltaOptimizer(learning_rate=50):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/AdadeltaOp_2.png)

#### AdagradOptimizer(learning_rate=0.05):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/AdagradOp_2.png)

#### AdamOptimizer(learning_rate=0.05):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/AdamOp_2.png)

#### FtrlOptimizer(learning_rate=0.05):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/FtrlOp_2.png)

#### GradientDescentOptimizer(learning_rate=0.05):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/GDOp_2.png)

#### MomentumOptimizer(learning_rate=0.05, momentum=0.9)
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/MomentumOp_2.png)

#### RMSPropOptimizer(learning_rate=0.05)
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/RMSPropOp_2.png)



#### AdadeltaOptimizer(learning_rate=1000):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/AdadeltaOp.png)

#### AdagradOptimizer(learning_rate=0.5):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/AdagradOp.png)

#### AdamOptimizer(learning_rate=0.5):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/AdamOp.png)

#### FtrlOptimizer(learning_rate=0.5):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/FtrlOp.png)

#### GradientDescentOptimizer(learning_rate=0.05):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/GDOp.png)

#### MomentumOptimizer(learning_rate=0.05, momentum=0.9)
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/MomentumOp.png)

#### ProximalAdagradOptimizer(learning_rate=0.5):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/ProximalAdagradOp.png)

#### ProximalGradientDescentOptimizer(learning_rate=0.05):
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/ProximalGDOp.png)

#### RMSPropOptimizer(learning_rate=0.5)
![](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/RMSPropOp.png)



#### Inspired by the following GIFs I found on the web:
![](https://i.stack.imgur.com/qAx2i.gif)
![](https://i.stack.imgur.com/1obtV.gif)