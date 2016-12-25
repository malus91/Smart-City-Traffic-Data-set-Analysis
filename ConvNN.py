from __future__ import print_function
import tensorflow as tf

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#filename = "data.csv"
filename = "final1.csv"
# setup text reader
file_length = file_len(filename)
filename_queue = tf.train.string_input_producer([filename])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

record_defaults = [tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.float32),tf.constant([], dtype=tf.float32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.int32),tf.constant([], dtype=tf.float32),tf.constant([], dtype=tf.int32)]

dayType,dayTime,longitude,latitude,ozone,particulateMatter,carbon_monoxide,sulfure_dioxide,nitrogen_dioxide,dew,hum,pressure,temp,wdir,wspeed,Trafficstatus = tf.decode_csv(
    value, record_defaults=record_defaults)
    
features = tf.pack([dayType,dayTime,tf.to_int32(longitude, name='ToInt32'),tf.to_int32(latitude, name='ToInt32'),ozone,particulateMatter,carbon_monoxide,sulfure_dioxide,nitrogen_dioxide,dew,hum,pressure,temp,wdir,tf.to_int32(wspeed, name='ToInt32')])

#A Convolutional Network implementation

learning_rate = 0.001
training_iters = 100000
#batch_size = 128
batch_size = 1000
display_step = 10

n_input = 800000
n_classes = 2
dropout = 0.75

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


#Create Model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = maxpool2d(conv2, k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
  


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

#Launch Graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        for i in range(file_length):
            batch_x, batch_y = sess.run([features, Trafficstatus])
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.})
            step += 1    
    print("Optimization Finished!")
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: features,y: Trafficstatus,keep_prob: 1.}))
    

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(file_length):
      example, label = sess.run([features, Trafficstatus])
      print(example, label)
    
