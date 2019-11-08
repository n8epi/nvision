"""
n-Vision

author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela

def nvis(data, res):
    '''
    Compute images and image embedding matrix
    :param data: numpy array of data in rows
    :param res: resolution of square images
    :return: images (res by res by x.shape[0] numpy array of images), embedding matrix
    '''

    n = res ** 2
    num_examples = data.shape[0]
    dim = data.shape[1]

    if num_examples > 10 ** 4 and dim < 10 ** 5:
        q = np.transpose(data) @ data
        s_x, v_x = la.eigh(q)
        v_x = np.fliplr(v_x)
    else:
        u_x, s_x, v_x = la.svd(data)  # initial x will be v @ stuff

    # Form the 1d graph Laplacian and eigendecomposition
    t = [2] * res
    t[0] = 1
    t[-1] = 1
    m = [-1] * (res - 1)
    a = sparse.diags([t, m, m], [0, -1, 1])
    w, v = sparsela.eigsh(a, k=(res - 1))

    # Do silly things because eigsh won't return all eigenvalues/eigenvectors
    w = np.concatenate((np.array([0]), w))
    v = np.concatenate((np.ones((res, 1)) / np.sqrt(res), v), axis=1)

    # Form the matrix of sums of all pairs which constitutes the eigenvalues of the graph Laplacian
    product_laplacian_eigen = np.add.outer(w, w)
    a, b = np.unravel_index(np.argsort(product_laplacian_eigen, axis=None)[:dim], (res, res))

    u = np.zeros((dim, n))
    for i in range(dim):
        u[i, :] = np.kron(v[:, a[i]], v[:, b[i]])

    f = v_x @ u # The embedding matrix
    ims = data @ f  # Matrix of unraveled images

    images = np.reshape(ims.T, (res, res, data.shape[0]))

    return images, f

def nvis_spectral(data, res):
    '''
    Compute images and image embedding matrix using the Schroedinger Laplacian
    :param data: numpy array of data in rows
    :param res: resolution of square images
    :return: images (res by res by x.shape[0] numpy array of images), embedding matrix
    '''

    n = res ** 2
    num_examples = data.shape[0]
    dim = data.shape[1]

    if num_examples > 10 ** 4 and dim < 10 ** 5:
        q = np.transpose(data) @ data
        s_x, v_x = la.eigh(q)
        v_x = np.fliplr(v_x)
    else:
        u_x, s_x, v_x = la.svd(data)  # initial x will be v @ stuff

    # Form the 1d graph Schroedinger Laplacian and eigendecomposition
    t = [2] * res
    m = [-1] * (res - 1)
    a = sparse.diags([t, m, m, [-1], [-1]], [0, -1, 1, 1-res, res-1])

    a = a + np.conjugate((np.fft.fft(np.conjugate(np.fft.fft(a.todense())).T)).T)/res
    w, v = la.eigh(a)

    # Form the matrix of sums of all pairs which constitutes the eigenvalues of the graph Laplacian
    product_laplacian_eigen = np.add.outer(w, w)
    a, b = np.unravel_index(np.argsort(product_laplacian_eigen, axis=None)[:dim], (res, res))

    u = 0.j*np.zeros((dim, n))
    for i in range(dim):
        u[i, :] = np.reshape(np.kron(v[:, a[i]], v[:, b[i]]),(n,))

    f = v_x @ u # The embedding matrix
    ims = data @ f  # Matrix of unraveled images

    images = np.reshape(ims.T, (res, res, data.shape[0]))

    return images, f
