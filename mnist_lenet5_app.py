#coding:utf-8
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import time

def restore(testPicArr):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32,[None,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
		y = mnist_lenet5_forward.forward(x,None,None)
		y_prediction = tf.argmax(y,1)

		variables_averages = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.M0VING_AVERAGE_DECAY)
		variables_to_restore = variables_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				preValue = sess.run(y_prediction,feed_dict={x:testPicArr})
				return preValue
			else:
				print("No model found!")
				return -1

def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28),Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 10
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255-im_arr[i][j]
			if im_arr[i][j] < threshold:
				im_arr[i][j] = 0
			else:
				im_arr[i][j] = 255
	nm_arr = im_arr.reshape([1,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.IMAGE_SIZE,mnist_lenet5_forward.NUM_CHANNELS])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr,1.0/255.0)

	return img_ready

def application():
#	testNum = input("Input the number of test pictures:")
#	for i in range(testNum):
	testNum = 10
	accuracy = 0
	r_num = 0
	to_num = 0
	for i in range(testNum):
		to_num = 0
		r_num = 0
		print("{}:".format(i))
		for j in range(11):
			to_num += 1
			testPic = "test_pic/{} ({}).jpg".format(i,j+1)
			testPicArr = pre_pic(testPic)
			preValue = restore(testPicArr)
			if preValue == i:
				r_num += 1
			print("The prediction number is:{}".format(preValue[0]))
			time.sleep(0.01)
		print("Accuracy is {}".format(r_num*1.0/to_num))
		accuracy += r_num*1.0/to_num
	accuracy = accuracy/10
	print("Total accuracy is {}".format(accuracy))

def main():
	application()

if __name__=='__main__':
	main()
