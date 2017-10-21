import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
def get_filequeue(files):
    filename=tf.train.string_input_producer(files,shuffle=False)
    reader=tf.WholeFileReader()
    key,image_buffer=reader.read(filename)
    im_tensor=tf.image.decode_jpeg(image_buffer,channels=3)
    im_tensor=tf.image.convert_image_dtype(im_tensor,dtype=tf.float32)
    im_tensor=tf.image.resize_images(im_tensor,(FLAGS.input_size,FLAGS.input_size),0)
    im_tensor=tf.reshape(im_tensor,shape=(1,FLAGS.input_size,FLAGS.input_size,3))
    im_tensor=tf.subtract(im_tensor,0.5)
    im_tensor=tf.multiply(im_tensor,5.0)
    return im_tensor
