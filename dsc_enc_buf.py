import numpy as np
from init_pps_params import initPps
PRINT_DEBUG_OPT = False
PRINT_FUNC_CALL_OPT = False
BUF_BIT_DEBUG_OPT = True

class DSCBuffer():

    def __init__(self, pps):
        self.buf_size = pps.chunk_size * pps.slice_height * 8 # in bit unit, not BYTE
        self.slices_per_line = int(pps.pic_width / pps.slice_width)

        ## Buffer Structure : (slices_per_line, buf_size) shape ndarray...
        # self.data = np.zeros((self.slices_per_line, self.buf_size), dtype = np.bool)
        self.data = np.zeros(self.buf_size * self.slices_per_line, dtype = np.bool)
        self.slice_index = 0 ## Value : [0, 1, 2, ... , slices_per_line]

        ###################
        # Write Data Like THIS:...
        # self.data[self.slice_index, bit_range_1 : bit_range_2] = Value_from_Addbits_function...
        ###################

        ## Number of bits read/written post-mux
        self.postMuxNumBits = 0 ## Pointer for data[i, :] Vector.
        # self.buf_size = (self.data).shape[0]

        self.BIT_DSC_PYTHON = open("BUF_BIT_DSC_PYTHON.txt", "wb")
        self.FIFO_DSC_PYTHON = open("SW_FIFO_DEBUG_PYTHON.txt", "w")

        return None

    def buf_reset(self):
        self.data[:, :] = 0
        # self.data[:] = 0
        self.postMuxNumBits = 0
        self.slice_index = 0

        return self


## Write a .DSC formatted file
def write_dsc_data(path, buf, pps):
    if PRINT_FUNC_CALL_OPT: print("currline_to_pic has called!!")
    slices_per_line = int(pps.pic_width / pps.slice_width)
    current_idx = np.zeros([slices_per_line, 1], dtype = np.int32)
    # total_byte = 0
    nbytes = pps.chunk_size ## Unit of BYTE!
    nbits = nbytes * 8 ## Buffer is an bool numpy array in python model...

    # with open(path, "ab") as f:
    #     for sy in range(pps.slice_height):
    #         for sx in range(slices_per_line):
    #             ## Lets Consider in BIT ORDER!!
    #             for i in range(nbytes):
    #
    #                 val = 0
    #
    #                 for j in range(i, i + 8):
    #                     # if (PRINT_DEBUG_OPT):
    #                     #     tmp = (current_idx[sx, :] + i + j).item()
    #                     #     print("WRITING VALUE : %x" % tmp)
    #
    #                     bit = int(buf.data[current_idx[sx, :] + i + j].item())
    #                     val = (val << 1) + bit
    #
    #                 if (PRINT_DEBUG_OPT):
    #                     print("WRITING VALUE : %x" %val)
    #
    #                 val = val.to_bytes(1, byteorder = 'big')
    #                 if (BUF_BIT_DEBUG_OPT): (buf.BIT_DSC_PYTHON).write(val)
    #                 f.write(val)
    #
    #             current_idx[sx] += nbytes

    with open(path, "ab") as f:
        for sy in range(pps.slice_height):
            for sx in range(slices_per_line):
                ## Lets Consider in BIT ORDER!!
                for i in range(0, nbits, 8):

                    val = 0

                    for j in range(i, i + 8):
                        # if (PRINT_DEBUG_OPT):
                        #     tmp = (current_idx[sx, :] + i + j).item()
                        #     print("WRITING VALUE : %x" % tmp)

                        bit = int(buf.data[current_idx[sx, :] + i + j].item())
                        val = (val << 1) + bit

                    if (PRINT_DEBUG_OPT):
                        print("WRITING VALUE : %x" % val)

                    val = val.to_bytes(1, byteorder = 'big')
                    if (BUF_BIT_DEBUG_OPT): (buf.BIT_DSC_PYTHON).write(val)
                    f.write(val)

                current_idx[sx] += nbytes

            total_byte += pps.chunk_size

    return total_byte
    return True
