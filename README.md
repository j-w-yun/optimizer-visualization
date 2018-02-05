# optimizer-visualization
Visualize loss minimization techniques in Tensorflow.

####Controls:
```
x_i = -1.8 // starting location x
y_i = 3.8 // starting location y
epoch=50 // iterations
learning_rate=0.5 // unless otherwise stated.
```

###AdadeltaOptimizer(learning_rate=1000):
learning rate of 0.5 was too small for AdadeltaOptimizer.
![1](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/AdadeltaOp.png)

###AdagradOptimizer(learning_rate=0.5):
![2](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/AdagradOp.png)

###AdamOptimizer(learning_rate=0.5):
![3](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/AdamOp.png)

###FtrlOptimizer(learning_rate=0.5):
![4](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/GDOp.png)

###GradientDescentOptimizer(learning_rate=0.05):
learning rate of 0.5 was too large for GDO.
![5](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/FtrlOp.png)

###MomentumOptimizer(learning_rate=0.05, momentum=0.9)
learning rate of 0.5 was too large for MomentumOptimizer.
![6](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/MomentumOp.png)

###ProximalAdagradOptimizer(learning_rate=0.5):
![7](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/ProximalAdagradOp.png)

###ProximalGradientDescentOptimizer(learning_rate=0.05):
learning rate of 0.5 was too large for ProximalGradientDescentOptimizer.
![8](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/ProximalGDOp.png)

###RMSPropOptimizer(learning_rate=0.5)
![9](https://github.com/Jaewan-Yun/optimizer-visualization/visuals/RMSPropOp.png)

###Inspired by the following GIFs I found on the web:

![10](https://i.stack.imgur.com/qAx2i.gif)
![11](https://i.stack.imgur.com/1obtV.gif)