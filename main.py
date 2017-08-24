import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag                    = 'vgg16'
    vgg_input_tensor_name      = 'image_input:0'
    vgg_keep_prob_tensor_name  = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # Extract layers of vgg 
    vgg_graph  = tf.get_default_graph()
    vgg_input  = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep   = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input, vgg_keep, vgg_layer3, vgg_layer4, vgg_layer7
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    vgg_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, name='vgg_layer7')

    vgg_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, name='vgg_layer4')

    vgg_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, name='vgg_layer3')
     
    # Add transpose layer                
    up_layer1 = tf.layers.conv2d_transpose(vgg_layer7, num_classes, kernel_size=4, strides=(2,2), padding='same', name='decoder_layer1')

    # Add skip layer
    skip1 = tf.add(vgg_layer4, up_layer1, name='skip_layer1')      

    # Add transpose layer                
    up_layer2 = tf.layers.conv2d_transpose(skip1, num_classes, kernel_size=4, strides=(2,2), padding='same', name='decoder_layer2')

    # Add skip layer
    skip2 = tf.add(up_layer2, vgg_layer3, name='skip_layer2')

    # Add transpose layer                
    out_layer = tf.layers.conv2d_transpose(skip2, num_classes, kernel_size=16, strides=(8,8), padding='same', name='decoder_layer3')      

    return out_layer

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # 4d to 2d
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    y = tf.reshape(correct_label, (-1, num_classes))

    # now define a loss function and a trainer/optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in tqdm(range(epochs)):
        train_loss  = 0
        num_samples = 0
        print("Train epoch: {}".format(i+1))

        # Train
        for X, y in get_batches_fn(batch_size):
                num_samples += len(X)
                loss, _ = sess.run(
                    [cross_entropy_loss, train_op],
                    feed_dict={input_image: X, correct_label: y, keep_prob: 0.8})
                train_loss += loss

        # Calc loss
        train_loss /= num_samples
        print("train_loss: {}".format(train_loss))

tests.test_train_nn(train_nn)

def run():
    # run parameters
    num_classes   = 2
    valid_size    = 0.2
    image_shape   = (160, 576)
    data_dir      = './data'
    runs_dir      = './runs'
    
    # training parameters
    epochs        = 25
    batch_size    = 1
    learning_rate = tf.constant(0.0001)
    train_network = False
    load_cp       = True
    augment_data  = False
    validate_nn   = True
    save_images   = False

    # Download kitti dataset
    helper.maybe_download_kitti(data_dir)
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Split training/validation data
    helper.split_test_validation(os.path.join(data_dir, 'data_road/training'), valid_size)

    # Augment training data
    if augment_data:
        helper.augment_data(os.path.join(data_dir, 'data_road/training/train'), image_shape)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training/train'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        vgg_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        out_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        logits, train_op, cross_entropy_loss = optimize(out_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        if train_network:
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                cross_entropy_loss, vgg_input, correct_label, keep_prob,
                learning_rate)
        elif load_cp:
            saver = tf.train.Saver()
            saver.restore(sess, 'runs/1503370650.1891687/model/model.ckpt')

        # Test neural network
        if validate_nn:
            helper.test_nn(os.path.join(data_dir, 
                'data_road/training/valid'), sess, 
                image_shape, logits, keep_prob, vgg_input, num_classes) 

        # TODO: Save inference data using helper.save_inference_samples
        if save_images:
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
