import os
import numpy as np
from init_enc_params import *

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

        #tmp = RESERVED.to_bytes(34, 'big')
        #f.write(tmp)

        ##########            Write DSC const                    ######

        qtc = dsc_const.quantTableChroma
        for i in range(5) :
            tmp = (qtc[i*6+0] + qtc[i*6+1] * 2 ** 5 + qtc[i*6+2] * 2 ** 10 + qtc[i*6+3] * 2 ** 15 +
                   qtc[i*6+4] * 2 ** 20 + qtc[i*6+5] * 2 ** 25).to_bytes(4, 'big')
            f.write(tmp)

        tmp = (qtc[30] + qtc[31] * 2 ** 5).to_bytes(4, 'big')
        f.write(tmp)

        qtl = dsc_const.quantTableLuma
        for i in range(5) :
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

def parse_pps(path = "w1.dsc", PRINT_PPS_OPT = False):
    PPS = initPps()

    with open(path, "rb") as f:
        if (f.read(4) != b'DSCF'):  # Magic Number
            raise ValueError("This file is not a DSC file.")

        tmp = int.from_bytes(f.read(1), byteorder = 'big', signed = 0)
        PPS.dsc_version_major = (tmp >> 4) & 0xf
        PPS.dsc_version_minor = tmp & 0xf

        PPS.pps_identifier = int.from_bytes(f.read(1), byteorder = 'big', signed = 0)  # PPS Identifier
        PPS.RESERVED = int.from_bytes(f.read(1), byteorder = 'big', signed = 0)  # RESERVED 8bits

        tmp = int.from_bytes(f.read(1), byteorder = 'big', signed = 0)
        PPS.bits_per_component = (tmp >> 4) & 0xf
        PPS.line_buf_depth = tmp & 0xf

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.block_pred_enable = (tmp >> 13) & 0b1
        PPS.convert_rgb = (tmp >> 12) & 0b1
        PPS.simple_422 = (tmp >> 11) & 0b1
        PPS.vbr_enable = (tmp >> 10) & 0b1
        PPS.bits_per_pixel = (tmp & 0b0000001111111111)

        PPS.pic_height = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.pic_width = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.slice_height = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.slice_width = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.chunk_size = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.initial_xmit_delay = (tmp & 0b0000001111111111)

        PPS.initial_dec_delay = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.initial_scale_value = (tmp & 0b0000000000111111)

        PPS.scale_increment_interval = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.scale_decrement_interval = (tmp & 0xfff)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.first_line_bpg_ofs = tmp & 0b0000000000011111

        PPS.nfl_bpg_offset = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.slice_bpg_offset = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.initlal_offset = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.final_offset = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.flatness_min_qp = (tmp & 0b0001111100000000) >> 8
        PPS.RESERVED = 0x0
        PPS.flatness_max_qp = (tmp & 0b0000000000011111)

        PPS.rc_model_size = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.rc_edge_factor = (tmp & 0b0000111100000000) >> 8
        PPS.RESERVED = 0x0
        PPS.rc_quant_icnr_limit0 = (tmp & 0b0000000000011111)

        tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.rc_quant_icnr_limit1 = (tmp & 0b0001111100000000) >> 8
        PPS.rc_tgt_offset_hi = (tmp & 0b0000000011110000) >> 4
        PPS.rc_tgt_offset_lo = (tmp & 0b0000000000001111)

        # rc_buf_thresh, (14, ) shape ndarray, with 6 bit fractional bits
        PPS.rc_buf_thresh = np.array([
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6,
            int.from_bytes(f.read(1), byteorder = 'big', signed = 0) << 6
        ])

        # rc_range_parameters, (15, 3) shape ndarray
        rc_range_parameters = np.zeros(shape = (15, 3), dtype = np.int32)
        for i in range(rc_range_parameters.shape[0]):
            tmp = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
            rc_range_parameters[i, :] = [tmp >> 11, (tmp >> 6) & 0b11111, tmp & 0b111111]
            if rc_range_parameters[i, 2] >= 32:
                rc_range_parameters[i, 2] -= 64

        # insert ndarray to dict PPS
        PPS.rc_range_parameters = rc_range_parameters

        tmp = int.from_bytes(f.read(1), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.native_420 = (tmp >> 1) & 0b1
        PPS.native_422 = tmp & 0b1

        tmp = int.from_bytes(f.read(1), byteorder = 'big', signed = 0)
        PPS.RESERVED = 0x0
        PPS.second_line_bpg_offset = tmp & 0b11111

        PPS.nsl_bpg_offset = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)
        PPS.second_line_ofs_adj = int.from_bytes(f.read(2), byteorder = 'big', signed = 0)

        # PPS.RESERVED = int.from_bytes(f.read(34), byteorder = 'big', signed = 0)

        for i in range(5):
            tmp = int.from_bytes(f.read(4), byteorder = 'big', signed = 0)


    if PRINT_PPS_OPT:
        for key, val in PPS.items():
            print("PPS : %s, Value : " %key, val)

    return PPS



