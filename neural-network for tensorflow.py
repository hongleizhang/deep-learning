#encoding:utf-8

#TensorFlow实现前馈神经网络

import tensorflow as tf #深度学习库
import numpy as np #矩阵操作库

#用TensorFlow来演示XOR门问题

# 网络结构：2维输入 --> 2维隐藏层 --> 1维输出
# 学习速率（learning rate）：0.0001

D_input  = 2
D_hidden = 2
D_label  = 1
lr = 1e-4

#进行对于x,y的占位操作，x为输入特征，y为label
x = tf.placeholder(tf.float32, [None, D_input], name="x") #name控制tensorboard生成图中的名字，方便debug
t = tf.placeholder(tf.float32, [None, D_label], name="t")

#隐藏层操作

# 初始化W
W_h1 = tf.Variable(tf.truncated_normal([D_input, D_hidden], stddev=0.1), name="W_h")
# 初始化b
b_h1 = tf.Variable(tf.constant(0.1, shape=[D_hidden]), name="b_h")
# 计算Wx+b
pre_act_h1 = tf.matmul(x, W_h1) + b_h1
# 计算a(Wx+b)
act_h1 = tf.nn.relu(pre_act_h1, name='act_h')


#输出层操作

W_o = tf.Variable(tf.truncated_normal([D_hidden, D_label], stddev=0.1), name="W_o")
b_o = tf.Variable(tf.constant(0.1, shape=[D_label]), name="b_o")
pre_act_o = tf.matmul(act_h1, W_o) + b_o
y = tf.nn.relu(pre_act_o, name='act_y')

#损失函数
loss = tf.reduce_mean((self.output-self.labels)**2)
#更新方法，除tf.train.AdamOptimizer外还有tf.train.RMSPropOptimizer等。默认推荐AdamOptimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#数据准备
#X和Y是4个数据的矩阵，X[i]和Y[i]的值始终对应。
#数据类型：用python使用tensorflow时，输入到网络中的训练数据需要以np.array的类型存在。并且要限制dtype为32bit以下。
#变量后跟着“.astype('float32')”总可以满足要求。
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]
X=np.array(X).astype('int16')
Y=np.array(Y).astype('int16')

#创建session
sess = tf.InteractiveSession()
#初始化权重
tf.initialize_all_variables().run()


#训练网络：有三种方法GD,SGD,batch-GD

#1.GD
T=10000 #训练几次epoch
for i in range(T):
  sess.run(train_step,feed_dict={x:X,t:Y})

#2.SGD
for i in range(T):
    for j in range(X.shape[0]): #X.shape[0]表示样本个数
        sess.run(train_step,feed_dict={x:X[j],t:Y[j]})

#3.bath-GD
#shuffle函数
def shufflelists(lists): #多个序列以相同顺序打乱
    ri=np.random.permutation(len(lists[1]))
    out=[]
    for l in lists:
        out.append(l[ri])
    return out

b_idx=0 #batch计数
b_size=2 #batch大小
for i in range(T):
#每次epoch都打乱顺序
    X,Y = shufflelists([X,Y])
    while batch_idx<=X.shape[0]:
        sess.run(train_step,feed_dict={x:X[b_idx:b_idx+b_size],t:Y[b_idx:b_idx+b_size]})
        b_idx+=b_size #更新batch计数