# This code performs Convolution
import numpy as np

# generating the image 'img' and kernel 'ker'
np.random.seed(42)
img = np.random.randint(0,10,size=(10,10),dtype=int)
img = np.asarray(img)
ker = np.random.randint(0,10,size=(3,3),dtype=int)
ker = np.asarray(ker)

# declaring the padding and stride
# size of convolution
# hor = ((W + 2P - F)/S ) + 1
# ver = ((W + 2P - F)/S ) + 1
padding = 0
stride = 1

# Apply padding if needed
# we create one new image 'img_z' of size (size_l,size_r) with zeros 
# then replace the necessary rows and columns with the image 'img'
if padding > 0:
  size_l = len(img[0]) + 2 * padding
  size_r = len(img) + 2 * padding
  img_z = np.zeros((size_l,size_r),dtype=int)
  
  img_z[padding:len(img)+padding:1,padding:len(img[0])+padding:1] = img
  img = img_z

hor = ( (len(img[0]) - len(ker[0])) // stride ) + 1
ver = ((len(img) - len(ker)) // stride) + 1

# declaring conv matrix with the (hor,ver) size
conv = np.zeros((hor,ver),dtype=int)

# This method performs element wise multiplication and addition
def simple_conv(A,B):
  res = 0
  for i in range(0,len(A)):
    for j in range(0,len(A[0])):
      res += A[i][j] * B[i][j]
  return res 

# These loops calls simple_conv method with two arguments
# one is the image with sliced rows and columns
# other is the kernel
for m in range(0,hor * stride,stride):
  for n in range(0,ver * stride,stride):
    row_len = m + len(ker[0])
    ver_len = n + len(ker)

    if row_len <= len(img) and ver_len <= len(img[0]):
      conv[m//stride][n//stride] = simple_conv(img[m:row_len, n:ver_len], ker)


'''
example:
img = [[6 3 7 4 6 9 2 6 7 4]
 [3 7 7 2 5 4 1 7 5 1]
 [4 0 9 5 8 0 9 2 6 3]
 [8 2 4 2 6 4 8 6 1 3]
 [8 1 9 8 9 4 1 3 6 7]
 [2 0 3 1 7 3 1 5 5 9]
 [3 5 1 9 1 9 3 7 6 8]
 [7 4 1 4 7 9 8 8 0 8]
 [6 8 7 0 7 7 2 0 7 2]
 [2 0 4 9 6 9 8 6 8 7]]
ker = [[1 0 6]
 [6 7 4]
 [2 7 5]]
conv = [[196 214 212 187 141 177 181 172]
 [155 144 210 178 175 211 186 113]
 [204 183 238 174 209 146 192 158]
 [142 141 234 219 180 121 127 202]
 [132 136 186 169 162 141 183 242]
 [124 114 183 218 211 250 194 231]
 [186 171 118 256 229 221 182 194]
 [157 198 214 240 269 217 146 216]]
we can also try with even and odd padding, even and odd stride
'''
