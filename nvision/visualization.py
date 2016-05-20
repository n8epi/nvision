import tensorflow as tf
import scipy as sp
import scipy.spatial as spat
import numpy as np
import matplotlib.pyplot as plt
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


    # Imaging function for data
    def im(dat):

        nc = 63;
        colors = cm.gray(np.linspace(0,1,nc+1))
        cs = np.empty((len(dat),4))
        for i in range(0,len(dat)):
            cs[i,:]=colors[np.int32(np.floor(nc*dat[i]))]

        plt.scatter(xtsne,ytsne,color=cs)
        plt.show()

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
    data = tf.transpose(mnist.test.images)
    #i = tf.placeholder(tf.int64,shape=[2])
    #i = tf.placeholder(tf.int32,shape=None)
    y = tf.slice(data,[0,0],[784,10000])

    with tf.Session() as sess:  # create a session to evaluate the symbolic expressions
        #print(sess.run(y))
        x=sess.run(y)
        #print(x);

    imager = nvis(x,res=1024)

    n=902
    dat = []
    for r in x:
        dat.append(r[n])

    im = imager(dat)

    #print(im)
    plt.imshow(im,interpolation='none',cmap='gray')
    plt.show()

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