from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#MNIST data has 3 parts: 55K data points for training, 10K data points for testing, and 5K data points for validation.
#data must be split.  Must have untouched data to measure how much we HAVE learned!


#each x is an image of a number. Image is flattened into 784 dimensional vector. It is represented in a 2-D tensor of floating point numbers.
x = tf.placeholder(tf.float32, [None, 784])

#Make model paramenters. Need Wights and biases. They are initialized as vectors full of zeroes. Later, these will "learn" so it doesnt really matter what they initialize as.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
#this line stores the correct answers
y_ = tf.placeholder(tf.float32,[None, 10])

#define the model
# y = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x,W) + b)

#to make a model "learn", we must define what it means to be good. OR what it means to be bad. It's called a 'loss'/'cost' or aka how far off it is.
#use cross-entropy function


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
train_steps = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#start a session and initialize the variables.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#we run training step 1000 times in batches of 100. It takes 100 random data points from the training set and run "train_step." This is called Stochastic training: using small batches of random data.

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

#now, let's see how well our model did
#this line is asking if our end result is equal to the correct answer. It generates a list of booleans. True = 1, False = 0. Then it finds what the percentage of True's there are.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accurage = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
