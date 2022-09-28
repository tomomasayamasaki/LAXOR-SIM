import numpy as np
import LAXOR_Sim.Config as config


class PE:

    def __init__(self):
        self.XNORout = 0
        self.weights = 0
        self.input = 0
        self.bias = 0
        self.flag = 1
        self.bitsize_pe = config.BIT_SIZE_PE



    def SetXNORWeights(self, subweights):
        self.weights = np.array(subweights) #1024bits



    def SetXNORInput(self, subinput):
        self.input = np.array(subinput) #1024bits



    def SetPopArray(self, array, mode='XNOR_output'):
        if mode == 'XNOR_output':
            self.XNORout = array
        elif mode == 'bias':
            self.bias = array
        else:
            print('SetPopArray@PE_XNOR_POP mode error')



    def _transferfunc(self, out):
        NumOne = out
        NumAll = self.bitsize_pe
        NumZero = NumAll - NumOne
        out2 = NumOne - NumZero
        return out2



    def RunXNOR(self):
        """
            Compute XNOR.
            self.out is 1024 bits
        """
        out = np.where(self.weights==self.input, 1, 0)
        self.SetPopArray(out,'XNOR_output')


    def RunPopcount(self):
        """
            Compute Popcount
            input for this is 1024bits
        """
        out = np.count_nonzero(self.XNORout == 1)
        out_tf = self._transferfunc(out)
        out_tf = float(out_tf) + self.bias
        return out_tf
