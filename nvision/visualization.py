import tensorflow as tf
import scipy as sp
import scipy.spatial as spat
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
from sklearn.manifold import TSNE

# MNIST dataset extraction
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def nvis(data,res=16):
    # Obtain embedding of points
    model = TSNE(n_components=2, random_state=0, verbose=1)
    z = model.fit_transform(data)

    xtsne = []
    ytsne = []
    for a in z:
        xtsne.append(a[0])
        ytsne.append(a[1])

    xlim = [min(xtsne), max(xtsne)]
    ylim = [min(ytsne), max(ytsne)]

    # Append corners to z
    zz=np.append(z,np.array([[xlim[0],ylim[0]],[xlim[0],ylim[1]],[xlim[1],ylim[0]],[xlim[1],ylim[1]]]),axis=0)

    tri = spat.Delaunay(zz)

    # Get barycentric coordinates of endpoints for transformation
    gridbar = np.empty((res,res,3))
    idx = np.empty((res,res),dtype=np.int32)
    dx = (xlim[1]-xlim[0])/(res+1)
    dy = (ylim[1]-ylim[0])/(res+1)
    for i in range(0,res):
        for j in range(0,res):
            pt = [xlim[0]+dx*(j+1),ylim[0]+dy*(i+1)];
            idx[i,j] = tri.find_simplex(pt)
            bar = tri.transform[idx[i,j], :2].dot(pt - tri.transform[idx[i,j], 2])
            bar = np.append(bar, [1 - bar.sum()], axis=0)
            #print(bar)
            gridbar[i,j,:] = bar

    simp = tri.simplices
    #print(gridbar)


    print('Constructing image visualization function...')

    # Imaging function for data
    def im(dat):


        #print("Constructing the image...")
        nc = 63;
        colors = cm.gray(np.linspace(0,1,nc+1))
        cs = np.empty((len(dat),4))
        for i in range(0,len(dat)):
            cs[i,:]=colors[np.int32(np.floor(nc*dat[i]))]



        #plt.scatter(xtsne,ytsne,color=cs)
        #plt.show()

        dd=np.append(dat,[0,0,0,0],axis=0) # Append zeros for the corners
        v = np.empty((res,res))

        for i in range(0,res):
            for j in range(0,res):
                # Note that we have to flip the vertical for proper image alignment
                #print(idx[i,j])
                #print(gridbar[i,j,:])
                #print(simp[idx[i,j]])
                v[res-i-1,j] = gridbar[i,j,0]*dd[simp[idx[i,j]][0]]+gridbar[i,j,1]*dd[simp[idx[i,j]][1]]+gridbar[i,j,2]*dd[simp[idx[i,j]][2]]

        return v

    return im




if __name__ == '__main__':

    res = 32 # Resolution of converted images

    ntrain = 55000
    ntest = 10000

    print('Generating data from tensorflow...')
    data = tf.transpose(mnist.train.images)

    y = tf.slice(data,[0,0],[784,ntrain])
    z = tf.slice(mnist.test.images,[0,0],[ntest,784])

    with tf.Session() as sess:  # create a session to evaluate the symbolic expressions
        x=sess.run(y)

    with tf.Session() as sess:
        xtest=sess.run(z)

    print('Generating image converter...')
    imager = nvis(x,res=res)

    # Tests for imager...
    #im = imager(x[:,900])
    #plt.imshow(im,interpolation='none',cmap='gray')
    #plt.show()

    try:
        temp_tensor = np.load('../data/mnist_vis_train_images.npy')
        test_tensor = np.load('../data/mnist_vis_test_images.npy')
    except FileNotFoundError:
        print("Initialization of tensors...")
        x = np.transpose(x) # Transpose for convenient slicing
        temp_tensor = np.empty((res,res,ntrain))
        test_tensor = np.empty((res, res, ntest))
        print("Populating training tensor...")
        for i in range(0,ntrain):
            temp_tensor[:,:,i] = imager(x[i,:])
        print("Populating test tensor...")
        for i in range(0,ntest):
            test_tensor[:,:,i] = imager(xtest[i,:])

        print("Saving numpy arrays...")
        np.save('../data/mnist_vis_train_images.npy',temp_tensor)
        np.save('../data/mnist_vis_test_images.npy',test_tensor)

    print("Conversion to tensorflow...")
    mnist_vis_train_images = tf.convert_to_tensor(temp_tensor)
    mnist_vis_test_images = tf.convert_to_tensor(test_tensor)

    #print(x)
    #model = TSNE(n_components=2, random_state=0)
    #z = model.fit_transform(x)
    #tri = spat.Delaunay(z)

    #xtsne = []
    #ytsne = []
    #im = []
    #n=900;
    #k=0;
    #for a in z:
    #    xtsne.append(a[0])
    #    ytsne.append(a[1])
    #    im.append(x[k][n])
    #    k=k+1

    #xlim = [min(xtsne),max(xtsne)]
    #ylim = [min(ytsne),max(ytsne)]

    #center = [(xlim[0]+xlim[1])/2,(ylim[0]+ylim[1])/2]

    #ix = tri.find_simplex(center)
    #print(ix)
    #print(tri.transform[ix,:2])
    #bar = tri.transform[ix,:2].dot(center - tri.transform[ix,2])
    #print(bar)
    #print(np.array([1-bar.sum()]))
    #print(np.append(bar,[1-bar.sum()],axis=0))

    #nc = 63;
    #colors = cm.rainbow(np.linspace(0,1,nc+1))
    #print(sum(im))
    #print(sum(np.floor(nc*im)))
    #cs = []
    #for x in im:
    #    cs.append(colors[np.int32(np.floor(nc*x))])

    #plt.scatter(xtsne,ytsne,color=cs)
    #plt.interactive(False)
    #plt.show(block=True)