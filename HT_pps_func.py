import os
import numpy as np
from init_enc_params import *

qlevel_luma_8bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
qlevel_chroma_8bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
qlevel_luma_10bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                     7, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
qlevel_chroma_10bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                       9, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
qlevel_luma_12bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                     7, 8, 8, 9, 9, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0]
qlevel_chroma_12bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                       9, 9, 10, 10, 11, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0]
qlevel_luma_14bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                     7, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 13, 0, 0, 0, 0]
qlevel_chroma_14bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                       9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 14, 0, 0, 0, 0]
qlevel_luma_16bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                     7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 15]
qlevel_chroma_16bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15,
                       16, 16, 16]

class initDefines:
    def __init__(self, pps):
        self.NUM_BUF_RANGES =          15
        self.NUM_COMPONENTS =          4
        self.MAX_UNITS_PER_GROUP =     4
        self.MAX_NUM_SSPS = 		   4   # Number of substreams (ie., substream processors)
        self.SAMPLES_PER_UNIT =        3
        self.MAX_PIXELS_PER_GROUP =    6
        self.GROUPS_PER_SUPERGROUP =   4
        self.BP_RANGE =                13 #  was 10, modified for high throughput test modes
        self.BP_SIZE = 				   3
        self.PRED_BLK_SIZE = 		   3
        self.ICH_BITS = 			   5
        self.ICH_SIZE =                1 << self.ICH_BITS
        self.ICH_PIXELS_ABOVE =        7
        self.ICH_LAMBDA = 			   4
        self.OFFSET_FRACTIONAL_BITS =  11
        self.PPS_SIZE = 		       128
        self.BP_EDGE_COUNT = 		   3
        self.BP_EDGE_STRENGTH =        32
        self.PADDING_LEFT =            1  # Pixels to pad line arrays to the left
        self.PADDING_RIGHT =           0  # Pixels to pad line arrays to the right
        self.RC_SCALE_BINARY_POINT =   3
        self.UNITS_PER_GROUP = 4 if pps.native_422 else 3
        self.SOMEWHAT_FLAT_QP_THRESH = pps.somewhat_flat_qp_thresh
        self.SOMEWHAT_FLAT_QP_DELTA =  pps.somewhat_flat_qp_delta
        self.VERY_FLAT_QP = 1 + (2 * (pps.bits_per_component - 8))
        if pps.native_422:
            self.OVERFLOW_AVOID_THRESHOLD = -224
        else:
            self.OVERFLOW_AVOID_THRESHOLD = -172

        self.LARGE_INT = 			  1 << 30

        self.PT_MAP = 0
        self.PT_LEFT = 1
        self.PT_BLOCK = 2
        self.MAX_SE_SIZE = 4 * pps.bits_per_component + 4


class initDscConstants:
    def __init__(self, pps, defines):
        if pps.bits_per_pixel >> 4 == 8:
            self.quantTableLuma = qlevel_luma_8bpc
            self.quantTableChroma = qlevel_chroma_8bpc
        elif pps.bits_per_pixel >> 4 == 10:
            self.quantTableLuma = qlevel_luma_10bpc
            self.quantTableChroma = qlevel_chroma_10bpc
        elif pps.bits_per_pixel >> 4 == 12:
            self.quantTableLuma = qlevel_luma_12bpc
            self.quantTableChroma = qlevel_chroma_12bpc
        elif pps.bits_per_pixel >> 4 == 14:
            self.quantTableLuma = qlevel_luma_14bpc
            self.quantTableChroma = qlevel_chroma_14bpc
        elif pps.bits_per_pixel >> 4 == 0:
            self.quantTableLuma = qlevel_luma_16bpc
            self.quantTableChroma = qlevel_chroma_16bpc
        else: # Error
            raise NotImplementedError

        if (pps.line_buf_depth == 0):
            self.lineBufDepth = 16
        else:
            self.lineBufDepth = pps.line_buf_depth

        self.full_ich_err_precision = 0 ## TODO check the variable

        if pps.native_420 or pps.native_422:
            self.sliceWidth = int(pps.slice_width / 2)
        else:
            self.sliceWidth = int(pps.slice_width)

        self.pixelsInGroup = 3

        self.ichIndexUnitMap = np.zeros(defines.MAX_PIXELS_PER_GROUP, )
        for i in range(defines.MAX_PIXELS_PER_GROUP):
            if (pps.native_422):
                self.ichIndexUnitMap[0] = 3 # put first ICH index with 2nd luma unit
                self.ichIndexUnitMap[1] = 1
                self.ichIndexUnitMap[2] = 2
                self.ichIndexUnitMap[3] = 0
            else:
                self.ichIndexUnitMap[i] = i % 3

        if pps.native_422:
            self.unitsPerGroup = 4
            self.numSsps = 4
            self.numComponents = 4
        else:
            self.unitsPerGroup = 3
            self.numSsps = 3
            self.numComponents = 3

        # range_ = np.zeros(self.NUM_COMPONENTS, )
        self.cpntBitDepth = np.zeros(defines.NUM_COMPONENTS, ).astype(np.int32)
        for i in range(defines.NUM_COMPONENTS):
            self.cpntBitDepth[i] = (pps.bits_per_component)
            diff_cond = (pps.convert_rgb & (i is not 0) & (i is not 3) & (pps.bits_per_component is not 0))  # 16 bpc condition

            if (diff_cond):
                self.cpntBitDepth[i] += 1
                # range_[i] *= 2

        self.maxSeSize = np.zeros(4, dtype = np.int32)

        if (pps.bits_per_component == 0):
            self.maxSeSize[0] = self.maxSeSize[1] = self.maxSeSize[2] = self.maxSeSize[3] = 64
        else:
            self.maxSeSize[0] = (pps.bits_per_component * 4) + 4
            self.maxSeSize[1] = (pps.bits_per_component + pps.convert_rgb) * 4
            self.maxSeSize[2] = (pps.bits_per_component + pps.convert_rgb) * 4
            self.maxSeSize[3] = (pps.bits_per_component)  * 4


def tb_pps(path = "pps_write_test.txt", pps = None, dsc_const = None, defines = None):
    RESERVED = 0
    with open(path, "wb") as f:
        # for val, length in zip(pps.values(), pps_size):
        #     if isinstance(val, int):
        #         f.write()
        #     else:
        #         val_flat = val.flatten()
        #         length_flat = length.flatten()
        #
        #         for val_, len_ in zip(val_flat, length_flat):
        #             f.write()

        ## pps Write START
        tmp = (pps.dsc_version_major * (2 ** 4) + pps.dsc_version_minor).to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.pps_identifier.to_bytes(1, 'big')
        f.write(tmp)

        tmp = RESERVED.to_bytes(1, 'big')
        f.write(tmp)

        tmp = (pps.bits_per_component * (2 ** 4) + pps.line_buf_depth).to_bytes(1, 'big')
        f.write(tmp)

        tmp = (pps.block_pred_enable * (2 ** 13) + pps.convert_rgb * (2 ** 12) + pps.simple_422 * (2 ** 11) + pps.vbr_enable * (2 ** 10) + pps.bits_per_pixel).to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.pic_height.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.pic_width.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.slice_height.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.slice_width.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.chunk_size.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.initial_xmit_delay.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.initial_dec_delay.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.initial_scale_value.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.scale_increment_interval.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.scale_decrement_interval.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.first_line_bpg_ofs.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.nfl_bpg_offset.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.slice_bpg_offset.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.initlal_offset.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.final_offset.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.flatness_min_qp.to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.flatness_max_qp.to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.rc_model_size.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.rc_edge_factor.to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.rc_quant_incr_limit0.to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.rc_quant_incr_limit1.to_bytes(1, 'big')
        f.write(tmp)

        tmp = (pps.rc_tgt_offset_hi * (2 ** 4) + pps.rc_tgt_offset_lo).to_bytes(1, 'big')
        f.write(tmp)

        for idx, val in enumerate(pps.rc_buf_thresh):
            tmp = (pps.rc_buf_thresh.item(idx)) >> 6
            #print("pps :", tmp >> 6)
            tmp = (tmp).to_bytes(1, 'big')
            f.write(tmp)

        for lists in pps.rc_range_parameters:
            if lists[2].item() >= 0:
                tmp = int((lists[0].item() * (2 ** 11)) + (lists[1].item() * (2 ** 6)) + lists[2].item()).to_bytes(2, 'big')

            else:
                tmp = int((lists[0].item() * (2 ** 11)) + (lists[1].item() * (2 ** 6)) + (lists[2].item() + 64)).to_bytes(2, 'big') # Convert Usigned 6-bit to Signed 6-bit
            f.write(tmp)

        tmp = (pps.native_420 * 2 + pps.native_422).to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.second_line_bpg_offset.to_bytes(1, 'big')
        f.write(tmp)

        tmp = pps.nsl_bpg_offset.to_bytes(2, 'big')
        f.write(tmp)

        tmp = pps.second_line_bpg_offset.to_bytes(2, 'big')
        f.write(tmp)

        tmp = RESERVED.to_bytes(2, 'big')
        f.write(tmp)

        ## 94 Bytes...

        ##########            Write DSC const                    ######

        qtc = dsc_const.quantTableChroma ## TODO : quantTableChroma has 16 elements
        for i in range(5):
            tmp = (qtc[i*6+0] + qtc[i*6+1] * 2 ** 5 + qtc[i*6+2] * 2 ** 10 + qtc[i*6+3] * 2 ** 15 +
                   qtc[i*6+4] * 2 ** 20 + qtc[i*6+5] * 2 ** 25).to_bytes(4, 'big')
            f.write(tmp)

        tmp = (qtc[30] + qtc[31] * 2 ** 5).to_bytes(4, 'big')
        f.write(tmp)

        qtl = dsc_const.quantTableLuma ## TODO : quantTableLuma has 16 elements
        for i in range(5):
            tmp = (qtl[i*6+0] + qtl[i*6+1] * 2 ** 5 + qtl[i*6+2] * 2 ** 10 + qtl[i*6+3] * 2 ** 15 +
                   qtl[i*6+4] * 2 ** 20 + qtl[i*6+5] * 2 ** 25).to_bytes(4, 'big')
            f.write(tmp)

        tmp = (qtl[30] + qtl[31] * 2 ** 5).to_bytes(4, 'big')
        f.write(tmp)

        cbd = dsc_const.cpntBitDepth.tolist()
        tmp = (cbd[0] + cbd[1] * 2 ** 5 + cbd[2] * 2 ** 10 + cbd[3] * 2 ** 15).to_bytes(4, 'big')
        f.write(tmp)

        mss = dsc_const.maxSeSize.tolist()
        tmp = (mss[0] + mss[1] * 2 ** 8 + mss[2] * 2 ** 16 + mss[3] * 2 ** 24).to_bytes(4, 'big')
        f.write(tmp)

        tmp = (dsc_const.numComponents + dsc_const.full_ich_err_precision * 2 ** 3 + pps.somewhat_flat_qp_thresh * 2 ** 4 +
               pps.somewhat_flat_qp_delta * 2 ** 9 + pps.flatness_det_thresh * 2 ** 14 + defines.VERY_FLAT_QP * 2 **24
               ).to_bytes(4, 'big')
        f.write(tmp)

        oat = (2 ** 10) + defines.OVERFLOW_AVOID_THRESHOLD # negative value
        tmp = (oat + pps.muxWordSize * (2 ** 10) + defines.MAX_SE_SIZE * (2 **18)).to_bytes(4, 'big')
        f.write(tmp)

        tmp = (pps.chunk_size * pps.slice_height * 8).to_bytes(4, 'big')
        f.write(tmp)
