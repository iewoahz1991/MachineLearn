from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

A = Image.open("6.son.png", "r")
output_path = r'.\Pic'
if not os.path.exists(output_path):
    os.makedirs(output_path)

def restore(sigma, u, v, K):
    m = len(u)
    n = len(v)
    sigma_k = np.zeros((m, n))
    for k in range(K+1):
        sigma_k[k][k] = sigma[k]
    a = np.dot(np.dot(u, sigma_k), v)
    a[a<0] = 0
    a[a>255] = 255
    return np.rint(a).astype('uint8') #取整

a = np.array(A)
K = 50
u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0]) #其中v_r已经转置
u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
plt.figure(figsize=(10, 10), facecolor='w')
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

for k in range(1, K+1):
    print(k)
    R = restore(sigma_r, u_r, v_r, k)
    G = restore(sigma_g, u_g, v_g, k)
    B = restore(sigma_b, u_b, v_b, k)
    I = np.stack((R, G, B), 2)
    Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
    if k <= 12:
        plt.subplot(3, 4, k)
        plt.imshow(I)
        plt.axis('off')
        plt.title(u'奇异值个数：%d' % k)
plt.suptitle(u'SVD与图像分解', fontsize=18)
plt.tight_layout(2)
plt.subplots_adjust(top=0.9)
plt.show()