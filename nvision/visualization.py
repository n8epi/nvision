from nvision import nvis, nvis_spectral
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.preprocessing as pproc
from sklearn import datasets
from colorsys import hls_to_rgb

import sys
sys.path.insert(0, '../')

'''
This is a quick test performed on the Wisconsin Breast Cancer dataset in sklearn.
'''

def colorize(z):
    '''
    Stolen from stack overflow -- colorize complex images
    :param z:
    :return:
    '''
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def display_200_images(im, title, a, b, save = False, im_dir= ''):
    '''
    Simple function for displaying 200 images
    :param im: res by res by num_images (>200) numpy array
    :param title: title for the figure
    :param a: lower color limit
    :param b: upper color limit
    :return: nothing...
    '''
    fig = plt.figure(figsize=(25, 8))
    #fig.suptitle(title)
    gs = gridspec.GridSpec(8, 25)
    gs.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.
    for i in range(0, 200):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(im[:, :, i], interpolation='nearest', clim=(a, b))

    if save:
        plt.savefig(
            im_dir + title.replace(' ', '') + '.eps',
            format='eps',
            bbox_inches='tight',
            dpi=200)
    else:
        plt.show()


def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)

    return c


def display_200_images_cmplx(R, I, title, res, save = False, im_dir= ''):
    '''
    Simple function for displaying 200 images
    :param im: res by res by num_images (>200) numpy array
    :param title: title for the figure
    :param a: lower color limit
    :param b: upper color limit
    :return: nothing...
    '''

    hres = np.int(res/2)
    fig = plt.figure(figsize=(25, 8))
    #fig.suptitle(title)
    gs = gridspec.GridSpec(8, 25)
    gs.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.
    for i in range(0, 200):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        rpart = R[:,:,i]
        rpart = np.roll(rpart, hres, axis=0)
        rpart = np.roll(rpart, hres, axis=1)
        ipart = I[:,:,i]
        ipart = np.roll(ipart, hres, axis=0)
        ipart = np.roll(ipart, hres, axis=1)
        #rgb = np.stack([rpart, 0.25*np.ones((res,res)), ipart],axis=-1)
        rgb = colorize(rpart+1.j*ipart)
        plt.imshow(rgb, interpolation='nearest')

    if save:
        plt.savefig(
            im_dir + title.replace(' ', '') + '.eps',
            format='eps',
            bbox_inches='tight',
            dpi=200)
    else:
        plt.show()

def display_100_images_cmplx(R, I, title, res, save = False, im_dir= ''):
    '''
    Simple function for displaying 100 images
    :param im: res by res by num_images (>200) numpy array
    :param title: title for the figure
    :param a: lower color limit
    :param b: upper color limit
    :return: nothing...
    '''

    hres = np.int(res/2)
    fig = plt.figure(figsize=(10, 10))
    #fig.suptitle(title)
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.
    for i in range(0, 100):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        rpart = R[:,:,i]
        rpart = np.roll(rpart, hres, axis=0)
        rpart = np.roll(rpart, hres, axis=1)
        ipart = I[:,:,i]
        ipart = np.roll(ipart, hres, axis=0)
        ipart = np.roll(ipart, hres, axis=1)
        #rgb = np.stack([rpart, 0.25*np.ones((res,res)), ipart],axis=-1)
        rgb = colorize(rpart+1.j*ipart)
        plt.imshow(rgb, interpolation='nearest')

    if save:
        plt.savefig(
            im_dir + title.replace(' ', '') + '.eps',
            format='eps',
            bbox_inches='tight',
            dpi=200)
    else:
        plt.show()

# Set data and image space parameters
dim = 30  # Dimension of the data
num_data = 569
res = 64  # Image resolution
n = res ** 2  # Total number of pixels

# Load the data, store as a matrix of rows, and scale the features
bcw = datasets.load_breast_cancer()  # Breast cancer data
db = bcw.data[:, 0:dim]
robust_scaler = pproc.RobustScaler()
robust_scaler.fit_transform(db)
db = robust_scaler.transform(db)

print('Computing images and isometric embedding...')
images, f = nvis_spectral(db, res)

save=False
im_dir = './'

print('Preparing plots of n-vision images...')
#a = np.percentile(images, 1)
#b = np.percentile(images, 99)
rt = pproc.QuantileTransformer(n_quantiles=16)
it = pproc.QuantileTransformer(n_quantiles=16)
#rt = pproc.RobustScaler()
#it = pproc.RobustScaler()
ims = np.ndarray.flatten(images)
R = np.reshape(np.real(ims),(n*num_data,1))
I = np.reshape(np.imag(ims),(n*num_data,1))
#rt.fit(R)
#it.fit(I)
#R = np.reshape(rt.transform(R), (res, res, num_data))
#I = np.reshape(it.transform(I), (res, res, num_data))
R = np.reshape(R, (res, res, num_data))
I = np.reshape(R, (res, res, num_data))

images_0R = R[:, :, bcw.target == 0]
images_0I = I[:, :, bcw.target == 0]
images_1R = R[:, :, bcw.target == 1]
images_1I = I[:, :, bcw.target == 1]

#images_0 = images[:, :, bcw.target == 0]
#images_1 = images[:, :, bcw.target == 1]
#display_200_images(images_1, 'Image Space Embeddings of Benign Tumor Data', a, b, save=save, im_dir=im_dir)
#display_200_images(images_0, 'Image Space Embeddings of Malignant Tumor Data', a, b, save=save, im_dir=im_dir)

display_100_images_cmplx(images_0R, images_0I, 'Image Space Embeddings of Benign Tumor Data', res, save=save, im_dir=im_dir)
display_100_images_cmplx(images_1R, images_1I, 'Image Space Embeddings of Malignant Tumor Data', res, save=save, im_dir=im_dir)