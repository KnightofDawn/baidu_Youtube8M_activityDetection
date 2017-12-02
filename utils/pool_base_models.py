'''
    Created by babiking on Dec.2nd @tucodec
    Class codebook -> Reference "Real-time foreground–background segmentation using codebook model."
'''


import numpy as np
import math

class _codebook(object):

    def __init__(self, epsilon):
        # epsilon: the threshold of codebook...
        self.epsilon = epsilon



    def _cal_colordist(self, targt, ref):
        '''
            Function:
                    _colordist
                    i.e. calculate the color distortion between target and reference RGB vector
            Input:
                    [1] <vector> targt: target RGB vector
                    [2] <vector> ref:   reference RGB vector
            Output:
                    <float> colordist
        '''
        colordist = 0

        # dim: dimension of color-space, e.g. dim_RGB=3
        dim = len(targt)

        # *: elementwise product
        # inp_norm: relative inner product between target and reference vector (square)
        inp_norm = np.power(np.dot(targt, ref),2) / np.sum(np.power(ref,2))

        # targt_norm: L2-norm of targt vector (square)
        targt_norm = np.sum(np.power(targt,2))

        colordist = math.sqrt( np.abs(targt_norm - inp_norm) )

        return colordist



    def _check_brightness(self, I, Imin, Imax):
        '''
            Function:
                    _cal_brightness
            Input:
                    [1] <int> I:    target intensity
                    [2] <int> Imin: low-bound of intensity   
                    [3] <int> Imax: upper-bound of intensity
            Output:
                    <bool> check_brightness
        '''
        brightness = 0

        alpha = 0.5
        beta  = 1.3

        if (np.empty(Imin) | np.empty(Imax)):
            return False

        I_low  = alpha*Imax
        I_high = min(Imin/alpha, beta*Imax)

        if (I>=I_low and I<=I_high):
            return True
        else:
            return False



    def _build_codebook(self, video):
        '''
            Function:
                    _build_codebook
                    i.e. build up a codebook for each pixel value in a video
            Input:
                    <array> video: [nframe, height, width, channel], 
                                   data format: uint8
                                   number of channel: for RGB, channel=3
            Output:
                    <cell> codebook
        '''

        nframe = np.size(video, axis=0)
        height = np.size(video, axis=1)
        width  = np.size(video, axis=2)


        # initial a [height x width] dimensional dictionary...
        codebook = {}

        # enumerate each pixel in a video...
        for n in range(nframe):
            for x in range(height):
                for y in range(width):

                    # target RGB vector...
                    targt = video[n, x, y, :]

                    # I: intensity of this pixel...
                    I = np.sum(np.power(targt,2))

                    indx = np.ravel_multi_index([x, y], [height, width])

                    # code: [codeLen x 9]
                    #   i.e. different pixels have different code lengths...
                    #        9-dimensional -> [Rc, Gc, Bc, Imin, Imax, frequency, lambda, t0, tn]
                    #           note: lambda: the Maximum Negative Run-Length
                    code = codebook[indx]

                    codeFound = True

                    codeLen = np.size(code, axis=0)
                    for iic in range(codeLen):
                        code_element = code[iic,:]
                        color = code_element[0:3]
                        Imin  = code_element[3]
                        Imax  = code_element[4]
                        freq  = code_element[5]
                        NRL   = code_element[6]

                        n_begin = code_element[7]
                        n_end   = code_element[8]

                        if ( self._cal_colordist(targt, color) and self._check_brightness(I, Imin, Imax) ):
                            # update code_element...
                            color = (freq*color + targt) / (freq + 1)
                            Imin  = min(I, Imin)
                            Imax  = max(I, Imax)
                            freq  = freq + 1
                            NRL   = max(NRL, n - n_end)

                            # begin and end time...
                            n_end = n

                            codebook[indx][iic, :] = [color, Imin, Imax, freq, NRL, n_begin, n_end]


                            codeFound = True
                            break

                    if (codeFound == False):
                        code[codeLen, :] = [targt, I, I, 1, n-1, n, n]
                        codebook[indx] = code

        for x in range(height):
            for y in range(width):

                indx = np.ravel_multi_index([x,y], [height, width])
                code = codebook[indx]

                codeLen = np.size(code, axis=0)

                pos = 0
                for iic in range(codeLen):
                    code_element = code[iic, :]

                    NRL     = code_element[6]
                    n_begin = code_element[7]
                    n_end   = code_element[8]

                    # maximum negative run-time length...
                    MNRL = max(NRL, nframe-n_begin+n_end-1)
                    code[iic, 6] = MNRL

                    if ( MNRL <= nframe/2 ):
                        code[pos, :] = code[iic, :]
                        pos = pos + 1

                    codebook[indx] = code

                if ( pos < codeLen ):
                    codebook[indx] = codebook[indx][0:pos, :]

        return codebook

