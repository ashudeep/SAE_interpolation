#import pickle
import display_network
from train_sparse_autoencoder_on_mnist import *
import numpy as np
#interpolate between pathces[:,one] and patches[:,two]
def raw_interpolation(one, two, count):
	images=np.empty([784,count])
	for ii in xrange(count):
		current_patch=((patches[:,one]*ii)+(patches[:,two]*(count-ii)))/count
		images[:,ii]=current_patch
	display_network.display_network('interpolations/raw_interpolation'+str(one)+'_'+str(two)+'.png',images)
	print "Figure written to 'raw_interpolation.png'"

def deep_interpolation(one,two,count,W1,W2,b1,b2):
	deep_image1=np.dot(W1,patches[:,one])
	deep_image2=np.dot(W1,patches[:,two])
	images=np.empty([784,count])
	for ii in xrange(count):
		deep_interpolation_img=((deep_image1*ii)+deep_image2*(count-ii))/count
		images[:,ii]=np.dot(W2,deep_interpolation_img)
	display_network.display_network('interpolations/deep_interpolation'+str(one)+'_'+str(two)+'.png',images)
	print "Figure Written to 'deep_interpolation.png'"

def demo(one,two,count,W1,W2,b1,b2):
	raw_interpolation(one,two,count)
	deep_interpolation(one,two,count,W1,W2,b1,b2)


if __name__ == '__main__':
	W1=np.load('W1.npy')
	W2=np.load('W2.npy')
	b1=np.load('b1.npy')
	b2=np.load('b2.npy')
	patches, _ = sample_images.get_mnist_data('../data/mnist.pkl.gz',
                                              train=True,
                                              num_samples=1000)
	one=126
	two=56
	demo(one,two,20,W1,W2,b1,b2)