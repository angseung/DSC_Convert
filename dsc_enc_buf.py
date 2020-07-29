import numpy as np
from init_pps_params import initPps

class DSCBuffer():

    def __init__(self, pps):
        buf_size = pps.chunk_size * pps.slice_height * 8 # in bit unit, not BYTE
        slices_per_line = int(pps.pic_width / pps.slice_width)
        self.data = np.zeros([slices_per_line, buf_size], dtype = np.bool)
        self.slice_index = 0
        self.postMuxNumBits = 0 ## Pointer for data[i, :] Vector.
        self.buf_size = self.data.shape[1]

        return None

    def buf_reset(self):
        self.data[:, :] = 0
        self.postMuxNumBits = 0
        self.slice_index = 0

        return self


def write_dsc_data(path, buf, pps):
    slices_per_line = int(pps.pic_width / pps.slice_width)
    current_idx = np.zeros([slices_per_line, 1])
    total_byte = 0
    with open("path", "ab") as f:
        for sy in range(pps.slice_height):
            for sx in range(pps.slice_width):
                for i in range(0, pps.chunk_size * 8, 8):

                    val = 0

                    for j in range(i + 8):
                        bit = buf.data[sx, current_idx[sx, 1] + i + j]
                        val = (val << 1) + bit

                    f.write(val)

            current_idx[sx] += pps.chunk_size
            total_byte += pps.chunk_size

    return total_byte
