import numpy as np
PRINT_DEBUG_OPT = 0

class DSCFifo:
    def __init__(self, size):
        self.data = np.zeros((8 * size, 1)).astype(np.int8) # this array can hold ("size") Bytes, and it can hold only bit '0' or '1'.
        self.size = 8 * size # in bit unit...
        self.fullness = 0
        self.read_ptr = 0
        self.write_prt = 0
        self.max_fullness = 0
        self.byte_ctr = 0

    def fifo_free(self):
        self.data[:, :] = 0

        return self

    def fifo_get_bits(self, n, sign_extend):
        if PRINT_DEBUG_OPT : print("FIFO_GET_BITS HAS BEEN CALLED")
        d = 0

        if (self.fullness < n):
            if PRINT_DEBUG_OPT:
                print("Fullness in Underflow Condition : %d when nbits is %d" %(self.fullness, n))
            raise ValueError("Fifo Underflow!")

        for i in range(n):
            b = int(self.data.item(self.read_ptr))

            if (i == 0) : sign = b # 'b' is a sign bit
            d = (d << 1) + b

            # if PRINT_DEBUG_OPT:
            #     print("b : ",b)
            #     print("d : ",d)

            self.fullness -= 1
            self.read_ptr += 1

            if (self.read_ptr >= self.size):
                self.read_ptr = 0

        if (sign_extend and sign):
            # mask = (1 << n) - 1
            # print("mask :", mask)
            d = d - (2 ** n)

        return d

    def fifo_put_bits(self, d, nbits):
        if PRINT_DEBUG_OPT: print("LETS WRITE %d BITS INTO FIFO, Value is %d" %(nbits, d))

        if (d.bit_length() > nbits):
            raise ValueError("Input Bit length is larger than 'nbit'")

        if (self.fullness + nbits > self.size):
            if PRINT_DEBUG_OPT:
                print("Fullness in Overflow Condition : %d when nbits is %d" %(self.fullness, nbits))
                print("FIFO MAXSIZE IS %d" %(self.size))
            raise ValueError("Fifo Overflowed!")

        self.fullness += nbits

        if (self.fullness > self.max_fullness):
            self.max_fullness = self.fullness

        for i in range(nbits):
            b = (d >> (nbits - i - 1)) & 1
            # if PRINT_DEBUG_OPT: print(b)

            self.data[self.write_prt] = b
            self.write_prt += 1

            if (self.write_prt >= self.size):
                self.write_prt = 0
