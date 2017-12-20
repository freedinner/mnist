import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True )
import tensorflow as tf

x =tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

"""在创建第一个卷积层之前，我们需要将输入数据x reshape成一个四维张量，
其中的2和3维对应着图像的weight和height最后一维对应着the number of color channels
(颜色通道数，因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)"""
x_image=tf.reshape(x,[-1,28,28,1])

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

"""stride size表示卷积步长，padding size表示边距"""
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

"""卷积在每个5x5的patch中算出32个特征卷积的权重张量形状是[5, 5, 1, 32]，
前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目"""
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
"""第一次卷积,28-5+1=24,图片变成24*24"""
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
"""第一次池化"""
h_pool1=max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
"""第二次卷积
"""
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
"""第二次池化"""
h_pool2 = max_pool_2x2(h_conv2)

"""现在图片的尺寸减小到7*7，再加入一个有1024个神经元的全连接层"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
"""第一个全连接层"""
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"""为了减少过拟合，我们在输出层之前加入dropout"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""输出层"""
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
"""ADAM优化器，代替梯度下降"""
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess=tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(200):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
      train_accuracy =sess.run(accuracy,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print ("test accuracy %g"% sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


