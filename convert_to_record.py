""" Converts a list of image files into a TFRecord using OpenCV.

This is particularly useful for image file formats that aren't supported
by native TensorFlow operations such as 'decode_jpeg' or 'decode_png'.
"""
import argparse
import os
import cv2
import logging
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True,
                        help="Text file with one image filename per line.")
    parser.add_argument("--output", help="Output filename", default=None)
    parser.add_argument("--size", nargs=2, type=int,
                        metavar=('WIDTH', 'HEIGHT'))

    args = parser.parse_args()
    if args.output is None:
        tokens = args.images.rsplit('.', 1)
        args.output = tokens[0] + ".tfrecord"

    if args.size is not None:
        args.size = tuple(args.size)

    return args

def image_generator(images_file):
    with open(images_file) as fp:
        for filename in fp:
            try:
                im = cv2.imread(filename.strip())
            except:
                logging.warning("Failed to read image: %s", filename)
                continue

            if im is not None:
                yield im

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_image(writer, image, size):
    # Note: Image size is (width, height) which is (shape[1], shape[0]) for
    # numpy-order arrays.
    image = cv2.resize(image, size)

    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
              }))

    # Write the Example proto to the TFRecord.
    writer.write(example.SerializeToString())

def read_and_decode_image(filename_queue, shape):
    """
    Args:
        shape: (width, height) of the image to decode
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    if len(shape) == 2:
        shape = (shape[1], shape[0], -1)

    row_major = tf.reshape(image, shape)
    return row_major

def create_tf_record(images, output_filename, size=None):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, im in enumerate(images):
        if size is None:
            size = im.shape[0:2]
            print "Detected image shape (%d x %d)" % (size[1], size[0])

        if idx % 1000 == 0 and idx != 0:
            print "Processed %d images" % idx

        encode_image(writer, im, (size[1], size[0]))

if __name__ == "__main__":
    args = get_args()
    create_tf_record(image_generator(args.images), args.output,
                     args.size)
