import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('./cameraman.png')
v_diff = img[1:,:,:] - img[:255,:,:]
h_diff = img[:,1:,:] - img[:,:255,:]
print(img.shape)
plt.subplot(1,3,1)
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(v_diff)
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(h_diff)
plt.axis('off')

plt.show()