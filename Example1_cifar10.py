####################################################
# Import package                                   #
####################################################
import numpy as np

# original
from LAXOR_Sim.Areca import Areca
import LAXOR_Sim.Tool as tool


####################################################
# Initilazation                                    #
####################################################
areca = Areca()


# input
input = np.random.randint(-1, 1, (1,3,32,32))

# bias
b1 = np.random.rand(64)
b2 = np.random.rand(128)
b3 = np.random.rand(256)
b4 = np.random.rand(512)
b5 = np.random.rand(10)

# weights
w1 = np.random.randint(-1, 1, (64,3,3,3))
w2 = np.random.randint(-1, 1, (64,64,3,3))
w3 = np.random.randint(-1, 1, (128,64,3,3))
w4 = np.random.randint(-1, 1, (128,128,3,3))
w5 = np.random.randint(-1, 1, (256,128,3,3))
w6 = np.random.randint(-1, 1, (256,256,3,3))
w7 = np.random.randint(-1, 1, (256,512))
w8 = np.random.randint(-1, 1, (512,512))
w9 = np.random.randint(-1, 1, (512,10))

# BNN
out = areca.CPU_Binary_Conv2D(input,w1,b1,1,0)
out = areca.Binary_Conv2D(out,w2,b1,1,0)
out = areca.MaxPooling(out,2,2)
out = areca.Binary_Conv2D(out,w3,b2,1,0)
out = areca.Binary_Conv2D(out,w4,b2,1,0)
out = areca.MaxPooling(out,2,2)
out = areca.Binary_Conv2D(out,w5,b3,1,0)
out = areca.Binary_Conv2D(out,w6,b3,1,0)
out = areca.Binary_FullyConnected(out, w7, b4)
out = areca.Binary_FullyConnected(out, w8, b4)
out = areca.Binary_FullyConnected(out, w9, b5)
