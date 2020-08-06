import numpy as np
import os
from PIL import Image
from dsc_fifo import *
PRINT_FUNC_CALL_OPT = False

def addbits(vlc_var, FIFO, data, nbits):
    if PRINT_FUNC_CALL_OPT: print("addbits has called!!")
    FIFO.fifo_put_bits(data, nbits)
    vlc_var.numBits += nbits

    return None

## A function to write bits to ENC buffer from FIFO
def putbits(val, size, buf):
    if PRINT_FUNC_CALL_OPT: print("putbits has called!!")

    ## BIT WRITE TEST WAS SUCCESSFUL
    if (size > 32):
        raise ValueError("error: putbits supports max of 32 bits")

    if (buf.postMuxNumBits > buf.buf_size):
        raise ValueError("Encoding Buffer Overflow!!")

    for i in range(size - 1, -1, -1):
        currbit = (val >> i) & 1
        ## DELETE THIS CHECK TO IMPROVE ENC SPEED
        # if (not ((currbit== 0) or (currbit == 1))):
        #     raise ValueError("Bit Value MUST BE bit 0 or bit 1")

        # print("Current postMuxNumBits : [%d]" %buf.postMuxNumBits)
        buf.data[buf.postMuxNumBits] = currbit

        ## TODO ENC BUFFER INDEX OUT OF BOUND FIX
        # try:
        #     buf.data[buf.postMuxNumBits] = currbit
        #
        # except :
        #     aa = 1

        buf.postMuxNumBits += 1

    return None


def getbits(size, buf, sign):
    if PRINT_FUNC_CALL_OPT: print("getbits has called!!")
    outval = 0

    for i in range(size):
        bit = buf.data[buf.slice_index, postMuxNumBits]
        if (i == 0): sign = bit

        outval = (outval << 1) + bit
        buf.postMuxNumBits += 1

        if (buf.postMuxNumBits > buf.buf_size):
            raise ValueError("Encoding Buffer Overflow!!")

    if (sign == 1): # Consider MSB sign bit
        outval = outval - (2 ** size)

    return outval


# b_num = '0bxxxxx' format
# returns integer format from binary format
def bin2dec(b_num):
    if PRINT_FUNC_CALL_OPT: print("bin2dec has called!!")
    b_num = list(b_num)
    value = 0
    for i in range(len(b_num)):
        digit = b_num.pop()
        if digit == '1':
            value = value + pow(2, i)
        elif digit ==  'b':
            value = value
        elif digit == '-':
            value = value * -1
    return value


# int_num = integer value
# returns shifted integer value
# shift_dir = the direction of shift operation
# example : 3 >> 2 is equal to bin_shift(3, 'right', 2)
def bin_shift(int_num, shift_dir = 'right', shift_len = 0):
    if PRINT_FUNC_CALL_OPT: print("bin_shift has called!!")
    b_num = bin(int_num)
    b_num = list(b_num)
    if shift_dir == 'right':
        for i in range(shift_len):
            b_num.pop()
    elif shift_dir == 'left':
        for i in range(shift_len):
            b_num.append(0)

    value = 0
    for i in range(len(b_num)):
        digit = b_num.pop()
        if digit == '1':
            value = value + pow(2, i)
        elif digit == 'b':
            value = value
        elif digit == '-':
            value = value * -1

    return value


def ceil_log2(val):
    if PRINT_FUNC_CALL_OPT: print("ceil_log2 has called!!")
    ret = 0
    x = val
    while (x):
        ret += 1
        x = x >> 1
    return ret


def FILT3(a,b,c):
    if PRINT_FUNC_CALL_OPT: print("FILT3 has called!!")
    return int((a + 2 * b + c + 2) >> 2)


def CLAMP(X, MIN, MAX):
    if PRINT_FUNC_CALL_OPT: print("CLAMP has called!!")
    if X > MAX:
        return MAX
    elif X < MIN:
        return MIN
    else:
        return X


def QuantDivisor(a):
    if PRINT_FUNC_CALL_OPT: print("QuantDivisor has called!!")
    arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    return arr[a]


def QuantOffset(a):
    if PRINT_FUNC_CALL_OPT: print("QuantOffset has called!!")
    arr = [0, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]
    return arr[a]

def rgb2ycocg(pps, im):
    im_yuv = np.zeros(im.shape, dtype = np.uint32)
    half = 1 << (pps.bitsPerPixel - 1)
    r = 0
    g = 1
    b = 2

    for xs in range(pps.pic_height): # 1080
        for ys in range(pps.pic_width): # 1920

            if (ys > 110):
                a = 100

            co = (im[xs, ys, r].item() - im[xs, ys, b].item())
            t = (im[xs, ys, b]).item() + (co >> 1)
            cg = (im[xs, ys, g]).item() - t
            y = t + (cg >> 1)

            if (pps.bitsPerPixel == 16):
                raise NotImplemented
            ### Pixel Value Debug
            if ((co >=256) or (cg >= 256) or (y >= 256)):
                a = 10

            else:
                co = co + 2 * half
                cg = cg + 2 * half

            im_yuv[xs, ys, :] = [y, co, cg]

            ## DEBUG ONLY
            # a = [y, co, cg]
            # a = 0

    return im_yuv