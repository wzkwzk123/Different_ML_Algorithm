import tensorflow as tf
import numpy as np



def save():
	## Save to file
	W = tf.Variable([[1,2,3], [4,5,6]], dtype=tf.float32, name="weights")
	b = tf.Variable([[1,2,3]],dtype=tf.float32, name="bias")

	init = tf.initialize_all_variables()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		save_path = saver.save(sess, "my_net/save_net.ckpt")
		print("save to path -- my_net/save_net.ckpt")


def restore():
	# define a empty framwork
	W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name="weights")
	b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name="bias")
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, "my_net/save_net.ckpt")
		print("weights:", sess.run(W))

restore()
