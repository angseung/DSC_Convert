import numpy as np
from init_pps_params import initPps
PRINT_DEBUG_OPT = 1
PRINT_FUNC_CALL_OPT = True

class DSCBuffer():

    def __init__(self, pps):
        buf_size = pps.chunk_size * pps.slice_height * 8 # in bit unit, not BYTE
        slices_per_line = int(pps.pic_width / pps.slice_width)
        # self.data = np.zeros([slices_per_line, buf_size], dtype = np.bool)
        self.data = np.zeros(buf_size * slices_per_line, dtype = np.bool)
        self.slice_index = 0

        ## Number of bits read/written post-mux
        self.postMuxNumBits = 0 ## Pointer for data[i, :] Vector.
        self.buf_size = (self.data).shape[0]

        return None

    def buf_reset(self):
        self.data[:] = 0
        self.postMuxNumBits = 0
        self.slice_index = 0

        return self


## Write a .DSC formatted file
def write_dsc_data(path, buf, pps):
    if PRINT_FUNC_CALL_OPT: print("currline_to_pic has called!!")
    slices_per_line = int(pps.pic_width / pps.slice_width)
    current_idx = np.zeros([slices_per_line, 1], dtype = np.int32)
    # total_byte = 0
    nbytes = pps.chunk_size
    nbits = nbytes * 8 ## Buffer is an bool numpy array in python model...

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
                        print("WRITING VALUE : %x" %val)

                    val = val.to_bytes(1, byteorder = 'big')
                    f.write(val)

                current_idx[sx] += nbytes

    #         total_byte += pps.chunk_size
    #
    # return total_byte
    return True
