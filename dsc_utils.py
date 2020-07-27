import numpy as np
import os
from PIL import Image
from dsc_fifo import *

def addbits(vlc_var, FIFO, data, nbits):
    FIFO.fifo_put_bits(data, nbits)
    vlc_var.numBits += nbits

    return None

def putbits(val, size, buf):
    if (size > 32):
        raise ValueError("error: putbits supports max of 32 bits")

    for i in range(size - 1, -1, -1):
        currbit = (val >> i) & 1
        if (currbit == 0 or currbit == 1):
            raise ValueError("Bit MUST BE bit 0 or bit 1")

        buf.data[buf.slice_index, buf.postMuxNumBits] = currbit
        buf.postMuxNumBits += 1

        if (buf.postMuxNumBits > buf.buf_size):
            raise ValueError("Encoding Buffer Overflow!!")

    return None

def getbits(size, buf, sign):
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


def ceil_log2(val) :
    ret = 0
    x = val
    while (x):
        ret += 1
        x = x >> 1
    return ret


def FILT3(a,b,c) :
    return (a + 2 * b + c + 2) / 2

def CLAMP(X, MIN, MAX):
    if X > MAX :
        return MAX
    elif X < MIN :
        return MIN
    else :
        return X

def QuantDivisor(a):
    arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    return arr[a]

def QuantOffset(a):
    arr = [0, 0, 1, 3,  7, 15, 31,  63, 127, 255,  511, 1023, 2047, 4095,  8191, 16383, 32767]
    return arr[a]
