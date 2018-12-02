# Load pickled data
import pickle
import matplotlib.pyplot as plt
#%matplotlib inline
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd

def one_hot_patch(x, depth):
    # workaround by name-name from github
    # NOTE: THIS CODE IS NOT BY ME. THERE ARE ISSUES WITH USING CERTAIN FUNCTION IN THE TENSORFLOW WINDOWS BUILD
    # SOURCE: https://github.com/tensorflow/tensorflow/issues/6509
    sparse_labels = tf.reshape(x, [-1, 1])
    derived_size = tf.shape(sparse_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(depth, [1])])
    return tf.sparse_to_dense(concated, outshape, 1.0, 0.0)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    out = []
    for i in range(len(img)):
        gray = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        out.append(cl1)
    out = np.array(out)
    out = out[..., np.newaxis]
    return out
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255


    image_data = a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
    return image_data


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # ADDING DROPOUT LAYER TO FIRST FEATURE MAP
    #conv1 = tf.nn.dropout(conv1, keep_prob)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))




    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))

    #fc0_drop = tf.nn.dropout(fc0, keep_prob)
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # ADDING DROPOUT LAYER TO FIRST FULLY CONNECTED LAYER
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    fc2_drop = tf.nn.dropout(fc2, keep_prob_2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2_drop, fc3_W) + fc3_b

    return logits

# TODO: Fill this in based on where you saved the training and testing data
EPOCHS = 30
BATCH_SIZE = 64

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, Y_train = train['features'], train['labels']
X_test, Y_test = test['features'], test['labels']
X_train, X_validation, Y_train, y_validation = train_test_split(X_train,Y_train,test_size=0.2, random_state=10)



### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)
n_validation = len(X_validation)


# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(Y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


#plt.imshow(image,cmap='gray')
#plt.show()
#print(Y_train[index])

n, bins, patches = plt.hist(Y_train, 42, normed=0, facecolor='green', alpha=0.75)
plt.title("Picture Label Histogram")
plt.xlabel("Label Number")
plt.ylabel("Frequency")

plt.show()


#Preproccessing to images here

index = random.randint(0,len(X_train))
image = X_train[index].squeeze()
plt.imshow(image)
plt.show()


X_train = grayscale(X_train)

image_g = X_train[index].squeeze()
plt.imshow(image_g,cmap='gray')
plt.show()

X_train = normalize_grayscale(X_train)


X_test = grayscale(X_test)
X_test = normalize_grayscale(X_test)

X_validation = grayscale(X_validation)
X_validation = normalize_grayscale(X_validation)

#X_train, Y_train = shuffle(X_train,Y_train)

##================================================

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
#x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)

one_hot_y = one_hot_patch(y, 43)

rate = 0.0015

logits = LeNet(x)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.9, keep_prob_2: 0.8})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, Y_train = shuffle(X_train, Y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.9, keep_prob_2: 0.8})

        validation_accuracy = evaluate(X_validation, y_validation)
        training_accuracy = evaluate(X_train, Y_train)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training  Accuracy = {:.3f}".format(training_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, Y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

print('Model Complete')



image_array =['Children crossing.jpg','Bumpy road.jpg','120 km per hr.jpg','30 km per hr.jpg','End of all speed and passing limits.jpg','Bicycles crossing.jpg']
classification_array = np.array([])

df = pd.read_csv('signnames.csv')
with tf.Session() as sess:
    pred = tf.nn.softmax(logits)
    sess.run(tf.global_variables_initializer())
    saver3 = tf.train.import_meta_graph('./lenet.meta')
    saver3.restore(sess, "./lenet")

    for image in image_array:
        image_test = cv2.imread(image)
        image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
        plt.imshow(image_test,cmap='gray')
        plt.show()
        image_test = cv2.resize(image_test, (32, 32))  # classifier takes 32X32 images
        image_test = np.array(image_test)
        resized_image = normalize_grayscale(image_test)
        resized_image = resized_image[..., np.newaxis]

        classification = sess.run(pred, feed_dict={x: [resized_image], keep_prob: 0.9, keep_prob_2: 0.8})
        classification_array = np.append(classification_array,np.argmax(classification))
        classification_5 = np.squeeze(classification)
        top_5 = classification_5.argsort()[-5:][::-1]
        top_5_percentage = np.sort(classification_5)[-5:][::-1]

        top_5_df = df.iloc[top_5]

        top_5_df = top_5_df.assign(Probability= pd.Series(top_5_percentage).values)

        print("Top 5 prediction for image {}".format(image))
        print(top_5_df)
        print()


print('finished')

