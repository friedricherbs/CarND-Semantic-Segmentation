import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from sklearn.metrics import jaccard_similarity_score
import cv2
from skimage import transform


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def maybe_download_kitti(data_dir):
    """
    Download and extract kitti road dataset if it doesn't exist
    :param data_dir: Directory to download the data to
    """
    kitti_filename = 'data_road.zip'
    kitti_path = os.path.join(data_dir, 'data_road')
   
    if not os.path.exists(kitti_path):
        # Clean kitti dir
        if os.path.exists(kitti_path):
            shutil.rmtree(kitti_path)
        os.makedirs(kitti_path)

        # Download kitti road dataset
        print('Downloading kitti road dataset...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'http://kitti.is.tue.mpg.de/kitti/data_road.zip',
                os.path.join(kitti_path, kitti_filename),
                pbar.hook)

        # Extract dataset
        print('Extracting dataset...')
        zip_ref = zipfile.ZipFile(os.path.join(kitti_path, kitti_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(kitti_path, kitti_filename))
    else:
        print('Found kitti dataset!')

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'images', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'labels', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    """
    Generate test output using the test images
    :param runs_dir: Directory to store the results to
    :param data_dir: Directory to download the data to
    :param sess: TF session
    :param image_shape: Tuple - Shape of image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param input_image: TF Placeholder for the image placeholder
    """
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    # Save graph
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tf.train.write_graph(sess.graph, model_dir, 'graph.pb', as_text=False)

    # Save weights
    saver = tf.train.Saver()
    model_path = os.path.join(model_dir, 'model.ckpt')
    save_path = saver.save(sess, model_path)
    print('Saved model to: {}'.format(save_path))

def augment_data(data_dir, image_shape):
    """Augument data by applying various image transformations
    """
    center     = (image_shape[1]/ 2, image_shape[0]/ 2)
    images     = glob(os.path.join(data_dir, 'images', '*.png'))
    num_images = len(images)

    # GT 
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_dir, 'labels', '*_road_*.png'))}

    print('Augmenting training data...')
    for i in tqdm(range(500)):
        rand_idx   = random.randint(0, num_images-1)
        image_file = images[rand_idx]
        image      = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        new_angle   = random.uniform(-15.0, 15.0)
        new_shift_x = random.uniform(-10.0, 10.0)
        new_shift_y = random.uniform(-10.0, 10.0)
        new_scale_x = random.uniform(0.9, 1.1)
        new_scale_y = random.uniform(0.9, 1.1)
        #new_shear   = random.uniform(-10.0/60, 10.0/60) #rad!

        
        M_rot     = cv2.getRotationMatrix2D(center,new_angle,1)
        M_shift   = np.float32([[1,0,new_shift_x],[0,1,new_shift_y]])
        M_scale   = np.float32([[new_scale_x,0,center[0]*(1.0-new_scale_x)],[0,new_scale_y,center[1]*(1.0-new_scale_y)]])
        #M_shear = transform.AffineTransform(shear=new_shear)
        
        new_image = cv2.warpAffine(image,     M_rot,  image_shape[::-1])
        new_image = cv2.warpAffine(new_image, M_scale,image_shape[::-1])
        new_image = cv2.warpAffine(new_image, M_shift,image_shape[::-1])
        #new_image = transform.warp(new_image, inverse_map=M_shear)

        # Maybe flip image vertically
        percentage_chance = 0.5
        flipped           = False
        if random.random() < percentage_chance:
            new_image = cv2.flip(new_image,1)
            flipped   = True

        # Save new image
        output_dir = os.path.join(data_dir, 'images')
        new_name   = 'augment_' + str(i) + '.png'
        scipy.misc.imsave(os.path.join(output_dir, new_name), new_image)

        # Get the corresponding gt data
        gt_image_file = label_paths[os.path.basename(image_file)]
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        new_gt = cv2.warpAffine(gt_image,  M_rot,  image_shape[::-1], borderValue=(255,0,0))
        new_gt = cv2.warpAffine(new_gt,    M_scale,image_shape[::-1], borderValue=(255,0,0))
        new_gt = cv2.warpAffine(new_gt,    M_shift,image_shape[::-1], borderValue=(255,0,0))

        if flipped:
            new_gt = cv2.flip(new_gt,1)

        # Save new gt image
        output_dir = os.path.join(data_dir, 'labels')
        new_name   = 'augment_road_' + str(i) + '.png'
        scipy.misc.imsave(os.path.join(output_dir, new_name), new_gt)

    print('Augmentation done!')


def split_test_validation(data_folder, test_size):
    """
    Split the training data into training and validation part
    :param data_folder: Path to the folder that contains the datasets
    :param test_size:   Fraction of training data which shall be used for validation
    """
    # Split training dataset to training and validation part
    print('Split training dataset in training and validation dataset ...')

    # Get paths
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

    # Select images for train and validation dataset
    n_all       = len(image_paths)
    all_idx     = list(range(n_all))

    random.seed(42)
    random.shuffle(all_idx)

    n_valid   = int(n_all*test_size)
    train_idx = all_idx[n_valid:]
    valid_idx = all_idx[:n_valid]

    # Create folder for training images
    output_dir_img = os.path.join(data_folder,     'train')
    output_dir_img = os.path.join(output_dir_img,  'images')
    if os.path.exists(output_dir_img):
        shutil.rmtree(output_dir_img)
    os.makedirs(output_dir_img)

    # Create folder for training labels
    output_dir_labels = os.path.join(data_folder,         'train')
    output_dir_labels = os.path.join(output_dir_labels,  'labels')
    if os.path.exists(output_dir_labels):
        shutil.rmtree(output_dir_labels)
    os.makedirs(output_dir_labels) 
    
    # Copy images and labels to new location
    for i in train_idx:
        src = image_paths[i]
        dst = os.path.join(output_dir_img, os.path.basename(src))
        shutil.copyfile(src, dst)

        gt_image_file = label_paths[os.path.basename(src)]
        src = gt_image_file
        dst = os.path.join(output_dir_labels, os.path.basename(src))
        shutil.copyfile(src, dst) 

    # Create folder for validation images
    output_dir_img = os.path.join(data_folder,     'valid')
    output_dir_img = os.path.join(output_dir_img,  'images')
    if os.path.exists(output_dir_img):
        shutil.rmtree(output_dir_img)
    os.makedirs(output_dir_img)

    # Create folder for training labels
    output_dir_labels = os.path.join(data_folder,        'valid')
    output_dir_labels = os.path.join(output_dir_labels,  'labels')
    if os.path.exists(output_dir_labels):
        shutil.rmtree(output_dir_labels)
    os.makedirs(output_dir_labels) 
    
    # Copy images and labels to new location
    for i in valid_idx:
        src = image_paths[i]
        dst = os.path.join(output_dir_img, os.path.basename(src))
        shutil.copyfile(src, dst)

        gt_image_file = label_paths[os.path.basename(src)]
        src = gt_image_file
        dst = os.path.join(output_dir_labels, os.path.basename(src))
        shutil.copyfile(src, dst) 

    print('Dataset split done!')

def test_nn(data_dir, sess, image_shape, logits, 
            keep_prob, image_pl, num_classes):
    """
    Calculate and print mean jaccard index to measure validation accuracy
    :param data_dir: Directory to download the data to
    :param sess: TF session
    :param image_shape: Tuple - Shape of image
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param num_classes: Number of classes for classification
    """

    all_iou = 0
    num_iou = 0
    label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_dir, 'labels', '*_road_*.png'))}

    bg_color = np.array([255, 0, 0])

    for image_file in glob(os.path.join(data_dir, 'images', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})

        # Get prediction from nn
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        
        # Get gt data
        gt_image_file = label_paths[os.path.basename(image_file)]
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        gt_road = np.all(gt_image == bg_color, axis=2)
        gt_road = np.logical_not(gt_road)

        # Calculate iou
        iou = jaccard_similarity_score(gt_road, np.squeeze(segmentation,axis=2))

        all_iou += iou
        num_iou += 1

    all_iou /= num_iou
    print("mean iou: {}".format(all_iou))
