import os
import numpy as np
from dsc_utils import *
from init_enc_params import initDefines, initFlatVariables, initDscConstants, initIchVariables, initPredVariables, initRcVariables, initVlcVariables

def isFlatnessInfoSent(pps, rc_var):
    is_flat_signaled = int((rc_var.masterQp >= pps.flatness_min_qp) & (rc_var.masterQp <= pps.flatness_max_qp))

    return is_flat_signaled


def isOrigFlatHIndex(hPos, currLine, rc_var, define, dsc_const, pps):
    fc1_start = 0
    fc1_end = 4
    fc2_start = 1
    fc2_end = 7
    # is_test1_successed = True
    # somewhat_flat = True
    # very_flat = True
    #
    # qp = max(rc_var.masterQp - define.SOMEWHAT_FLAT_QP_DELTA, 0)
    # thresh = np.zeros(define.NUM_COMPONENTS)
    #
    # for i in range(define.NUM_COMPONENTS):
    #     thresh[i] = MapQpToQlevel(pps, dsc_const, i, qp)
    #
    # is_end_of_slice = (hPos + 1) >= pps.slice_width
    # #return 0
    #
    # for cpnt in range(define.NUM_COMPONENTS):
    #     vf_thresh = 2
    #     max_val = -1
    #     min_val = 99999
    #
    #     for i in range(fc1_start, fc1_end):
    #         pixel_val = currLine[cpnt, define.PADDING_LEFT + hPos + 1]
    #
    #         if max_val < pixel_val: max_val = pixel_val
    #         if min_val > pixel_val: min_val = pixel_val
    #
    #     is_somewhatflat_falied = (max - min) > max(vf_thresh, QuantDivisor(thresh[cpnt]))
    #     is_veryflat_failed = (max - min) > vf_thresh
    #
    #     if not is_somewhatflat_falied:
    #         is_test1_successed = False
    #         somewhat_flat = False
    #
    #     if not is_veryflat_failed:
    #         is_test1_successed = False
    #         very_flat = False
    #
    # #### Flat Test 2
    #
    # for cpnt in range(define.NUM_COMPONENTS):
    #     vf_thresh = 2
    #     max_val = -1
    #     min_val = 99999
    #
    #     for i in range(fc2_start, fc2_end):
    #         pixel_val = currLine[cpnt, define.PADDING_LEFT + hPos + 1]
    #
    #         if max_val < pixel_val: max_val = pixel_val
    #         if min_val > pixel_val: min_val = pixel_val
    #
    #     is_somewhatflat_falied = (max - min) > max(vf_thresh, QuantDivisor(thresh[cpnt]))
    #     is_veryflat_failed = (max - min) > vf_thresh
    #
    #     if not is_somewhatflat_falied:
    #         is_test1_successed = False
    #         somewhat_flat = False
    #
    #     if not is_veryflat_failed:
    #         is_test1_successed = False
    #         very_flat = False
    #
    #


def flatnessAdjustment(hPos, groupCount, pps, rc_var, flat_var, define):
    cond = isFlatnessInfoSent(pps, rc_var) & (groupCount % define.GROUPS_PER_SUPERGROUP == 3)

    # if cond:
    #     flat_var.prevFirstFlat = -1
    #
    #     if flat_var.firstFlat >= 0:
    #         flat_var.prevIsFlat = 1
    #     else:
    #         flat_var.prevIsFlat = 0
    #
    #     for i in range(define.GROUPS_PER_SUPERGROUP):
    #         flatness_type = 0
    #

def calc_fullness_offset(vPos, pixelCount, groupCnt, pps, define, dsc_const, vlc_var, rc_var):
    unity_scale = define.RC_SCALE_BINARY_POINT
    throttleFrac = 0

    if groupCnt == 0:
        rc_var.currentScale = pps.initial_scale_value
        rc_var.scaleAdjustCounter = 1

    elif ((vPos == 0) and (rc_var.currentScale > unity_scale)):
        rc_var.scaleAdjustCounter += 1

        if (rc_var.scaleAdjustCounter > pps.scale_decrement_interval):
            rc_var.scaleAdjustCounter = 0
            rc_var.currentScale -= 1

    elif rc_var.scaleIncrementStart:
        rc_var.scaleAdjustCounter += 1

        if (rc_var.scaleAdjustCounter > pps.scale_increment_interval):
            rc_var.scaleAdjustCounter = 0
            rc_var.currentScale += 1

    if (vPos == 0):
        current_bpg_target = pps.first_line_bpg_ofs
        increment = - (pps.first_line_bpg_ofs << define.OFFSET_FRACTIONAL_BITS)

    else:
        current_bpg_target = pps.nfl_bpg_offset
        increment = pps.nfl_bpg_offset

    if (vPos == 1):
        current_bpg_target += pps.second_line_bpg_ofs
        increment += -(pps.second_line_bpg_ofs << define.OFFSET_FRACTIONAL_BITS)

        if (not rc_var.secondOffsetApplied):
            rc_var.secondOffsetApplied = 1
            rc_var.rcXformOffset -= pps.second_line_offset_adj

    else:
        cond = pps.scale_increment_interval and (not rc_var.scaleIncrementStart) and (vPos > 0) and (rc_var.rcXformOffset > 0)
        if cond:
            rc_var.currentScale = 9
            rc_var.scaleAdjustCounter = 0
            rc_var.scaleIncrementStart = 1

    rc_var.prevPixelCount = pixelCount
    current_bpg_target -= pps.slice_bpg_offset >> define.OFFSET_FRACTIONAL_BITS
    increment += pps.slice_bpg_offset
    throttleFrac += increment
    rc_var.rcXformOffset += throttleFrac
    throttleFrac = throttleFrac & ((1 << define.OFFSET_FRACTIONAL_BITS) - 1)

    if (rc_var.rcXformOffset < pps.final_offset):
        rc_var.rcOffsetClampEnable = 1

    if rc_var.rcOffsetClampEnable:
        rc_var.rcXformOffset = min(rc_var.rcXformOffset, pps.final_offset)

    return [rc_var.currentScale, rc_var.rcXformOffset]


def rate_control(vPos, pixelCount, sampModCnt, pps, ich_var, vlc_var, rc_var, flat_var, define):
    ## prev_fullness moved to main
    prev_fullness = rc_var.bufferFullness
    mpsel = (vlc_var.midpointSelected).sum()

    # pixelCount moved to enc_main
    # for i in range(sampModCnt):
    #     ### pixelCount???
    #     pass

    rcSizeGroup = 0
    for i in range(3):
        rcSizeGroup += vlc_var.rcSizeUnit[i]

    range_cfg = []

    ## Linear Transformation
    throttle_offset = rc_var.rcXformOffset
    throttle_offset -= pps.rc_model_size
    rcBufferFullness = (rc_var.currentScale * (rc_var.bufferFullness + rc_var.rcXformOffset)) >> define.RC_SCALE_BINARY_POINT

    for i in range(define.NUM_BUF_RANGES - 1, -1, 0):
        overflowAvoid = (rc_var.bufferFullness + rc_var.rcXformOffset) > define.OVERFLOW_AVOID_THRESHOLD

        if ((rcBufferFullness > pps.rc_buf_thresh[i - 1] - pps.rc_model_size)):
            break

    if (rcBufferFullness > 0):
        raise ValueError("The RC model has overflowed.")

    selected_range = rc_var.prevRange
    rc_var.prevRange = i

    bpg = (pps.bits_per_pixel * sampModCnt + 8) >> 4 #ROunding fractional bits
    rcTgtBitGroup = max(0, bpg + pps.rc_range_parameters[selected_range][2] + rc_var.rcXformOffset)
    min_QP = pps.rc_range_parameters[selected_range][0]
    max_QP = pps.rc_range_parameters[selected_range][1]
    tgtMinusOffset = max(0, rcTgtBitGroup - pps.rc_tgt_offset_lo)
    tgtPlusOffset = max(0, rcTgtBitGroup + pps.rc_tgt_offset_hi)
    incr_amount = (vlc_var.codedGroupSize - rcTgtBitGroup) >> 1

    ### How about make this param canstant??
    ### SW
    if pps.native_420:
        predActivity = rc_var.prevQp + max(vlc_var.predictedSize[0], vlc_var.predictedSize[1]) + vlc_var.predictedSize[2]
    elif pps.native_422:
        predActivity = rc_var.prevQp + ((predictedSize.sum()) >> 1)
    else: #444 Mode
        predActivity = rc_var.prevQp + vlc_var.predictedSize[0] + max(vlc_var.predictedSize[1], vlc_var.predictedSize[2])

    bitSaveThresh = define.cpntBitDepth[0] + define.cpntBitDepth[1] - 2

    ## bitSaveMode Decision Start...
    tmp_mppState = rc_var.mppState + 1
    bs_cond1 = (vPos > 0) & (flat_var.firstFlat == -1)
    bs_cond2 = (tmp_mppState >= 2)
    bs_cond3 = ((not ich_var.ichSelected) & (mpsel >= 3))
    bs_cond4 = ((not ich_var.ichSelected) & (predActivity >= bitSaveThresh))
    bs_cond5 = ich_var.ichSelected

    bs_case1 = bs_cond1 & bs_cond3 & bs_cond2
    bs_case2 = (bs_cond1 & bs_cond3 & (not bs_cond2))
    bs_case3 = (bs_cond1 & bs_cond4)
    bs_case4 = (bs_cond1 & bs_cond5)
    bs_case5 = (bs_cond1 & (not bs_cond5))
    bs_case6 = (not bs_cond1)

    if bs_case1:
        rc_var.bitSaveMode = 2
        rc_var.mppState += 1

    elif bs_case2:
        rc_var.mppState += 1

    elif bs_case3:
        rc_var.bitSaveMode = rc_var.bitSaveMode

    elif bs_case5:
        rc_var.mppState = 0
        rc_var.bitSaveMode = 0

    elif bs_case6:
        rc_var.mppState = 0
        rc_var.bitSaveMode = 0

    # if cond1:
    #     if (not ich_var.ichSelected) & (mpsel >= 3):
    #         rc_var.mppState += 1
    #         if (rc_var.mppState >= 2):
    #             rc_var.bitSaveMode = 2
    #
    #     elif (not ich_var.ichSelected) & (predActivity >= bitSaveThresh):
    #         rc_var.bitSaveMode = rc_var.bitSaveMode
    #     elif ich_var.ichSelected:
    #         rc_var.bitSaveMode = max(1, rc_var.bitSaveMode)
    #     else:
    #         rc_var.mppState = 0
    #         rc_var.bitSaveMode = 0
    #
    # else:
    #     rc_var.mppState = 0
    #     rc_var.bitSaveMode = 0

    ## Short-Term QP Adjustment Start...
    ## make condition to implement switch-case method
    ######### stqp Condition decision..######
    cond1 = overflowAvoid
    cond2 = (rc_var.bufferFullness <= 192)
    cond3 = rc_var.bitSaveMode == 2
    cond4 = rc_var.bitSaveMode == 1
    cond5 = (rc_var.rcSizeGroup == define.UNITS_PER_GROUP)
    cond6 = (rc_var.rcSizeGroup < tgtMinusOffset)
             #& (vlc_var.codedGroupSize < tgtMinusOffset))
    cond7 = ((rc_var.bufferFullness >= 64) &
             (vlc_var.codedGroupSize > tgtPlusOffset))
    cond8 = not cond7
    ##########################################

    if cond2: #underflow Condition
        rc_var.stQp = min_QP # cond2

    elif cond3:
        max_QP = min(pps.bits_per_component * 2 - 1, max_QP + 1)
        rc_var.stQp = rc_var.prevQp + 2  # cond3

    elif cond4:
        max_QP = min(pps.bits_per_component * 2 - 1, max_QP + 1)
        rc_var.stQp = rc_var.prevQp  # cond4

    elif cond5:
        min_QP = max(min_QP - 4, 0)
        rc_var.stQp = rc_var.prevQp - 1 # cond5

    elif cond6:
        rc_var.stQp = rc_var.prevQp - 1

    elif cond7:
        curQp = max(rc_var.prevQp, min_QP)

        inc_cond1 = (curQp == rc_var.prev2Qp)
        inc_cond2 = ((rc_var.rcSizeGroup * 2) < (rc_var.rcSizeGroupPrev * pps.rc_edge_factor))
        inc_cond3 = (rc_var.prev2Qp < curQp)
        inc_cond4 = (((rc_var.rcSizeGroup * 2) < (rc_var.rcSizeGroupPrev * pps.rc_edge_factor)) &
                     (curQp < pps.rc_quant_incr_limit0))
        inc_cond5 = (curQp < pps.rc_quant_incr_limit1)

        case1 = (inc_cond1 & inc_cond2)
        case2 = (inc_cond1 & (not inc_cond2))
        case3 = ((not inc_cond1) & inc_cond3 & inc_cond4)
        case4 = ((not inc_cond1) & inc_cond3 & (not inc_cond4))
        case5 = ((not inc_cond1) & (not inc_cond3) & inc_cond5)
        case6 = ((not inc_cond1) & (not inc_cond3) & (not inc_cond5))

        if (case1 or case3 or case5): rc_var.stQp = curQp + incr_amount
        if (case2 or case4 or case6): rc_var.stQp = curQp
        #
        #
        # if (curQp == rc_var.prev2Qp):
        #     cond = ((rc_var.rcSizeGroup * 2) < (rc_var.rcSizeGroupPrev * pps.rc_edge_factor))
        #     if cond:
        #         rc_var.stQp = curQp + incr_amount
        #     else:
        #         rc_var.stQp = curQp
        #
        # elif (rc_var.prev2Qp < curQp):
        #     cond = (((rc_var.rcSizeGroup * 2) < (rc_var.rcSizeGroupPrev * pps.rc_edge_factor)) &
        #             curQp < pps.rc_quant_incr_limit0)
        #     if cond:
        #         rc_var.stQp = curQp + incr_amount
        #     else:
        #         rc_var.stQp = curQp
        #
        # elif (curQp < pps.rc_quant_incr_limit1):
        #     rc_var.stQp = curQp + incr_amount
        #
        # else:
        #     rc_var.stQp = curQp

    elif cond8:
        rc_var.stQp = rc_var.prevQp

    elif cond1: # overflow avoid condition
        rc_var.stQp = pps.rc_range_parameters[define.NUM_BUF_RANGES - 1][0]  # cond1
        # max_QP = pps.rc_range_parameters[define.NUM_BUF_RANGES - 1][1]

    #
    # do_increment_logic = (rc_var.bufferFullness >= 64) & \
    #                      (vlc_var.codedGroupSize > tgtPlusOffset) # avoid increasing QP immediately after edge


    rc_var.stQp = CLAMP(rc_var.stQp, min_QP, max_QP)

    rc_var.rcSizeGroupPrev = rc_var.rcSizeGroup

    ## check rc buffer overflow
    is_overflowed = (rc_var.bufferFullness > pps.rcb_bits)

    if is_overflowed:
        raise ValueError("The buffer model has overflowed.")

    rc_var.masterQp = rc_var.prevQp

    return rc_var.masterQp


def FindResidualSize(eq):
    if (eq == 0): size_e = 0
    elif (-1 <= eq <= 0): size_e = 1
    elif (-2 <= eq <= 1): size_e = 2
    elif (-4 <= eq <= 3): size_e = 3
    elif (-8 <= eq <= 7): size_e = 4
    elif (-16 <= eq <= 15): size_e = 5
    elif (-32 <= eq <= 31): size_e = 6
    elif (-64 <= eq <= 63): size_e = 7
    elif (-128 <= eq <= 127): size_e = 8
    elif (-256 <= eq <= 255): size_e = 9
    elif (-512 <= eq <= 511): size_e = 10
    elif (-1024 <= eq <= 1023): size_e = 11
    elif (-2048 <= eq <= 2047): size_e = 12
    elif (-4096 <= eq <= 4095): size_e = 13
    elif (-8192 <= eq <= 8191): size_e = 14
    elif (-16384 <= eq <= 16383): size_e = 15
    elif (-32768 <= eq <= 32767): size_e = 16
    elif (-65536 <= eq <= 65535): size_e = 17
    elif (-131702 <= eq <= 131701): size_e = 18
    else:
        print("unexpectedly large residual size")
        raise ValueError
    return size_e


def MaxResidualSize(pps, dsc_const, cpnt, qp) :
    """
    :param pps: is_native_420, is_dsc_version_minor
    :param dsc_const: cpntBitDepth[cpnt], quantTableLuma[qp], quantTableChroma[qp]
    :return: qlevel
    """
    qlevel = MapQpToQlevel(pps, dsc_const, cpnt, qp)
    return dsc_const.cpntBitDepth[cpnt] - qlevel


def FindMidpoint(dsc_const, cpnt, qlevel, recon_value):
    """
    :param cpntBitDepth[cpnt]:
    :param qlevel:
    :param recon_value
    :return:
    """
    midrange = 1 << dsc_const.cpntBitDepth[cpnt]
    midrange = midrange / 2
    return (midrange + (recon_value % (1 << qlevel)))


def QuantizeResidual(err, qlevel):
    """
    :param err:
    :param qlevel:
    :return:
    """
    if err > 0:
        eq = (err + QuantOffset(qlevel)) >> qlevel
    else :
        eq = -1 * ((QuantOffset(qlevel) - err) >> qlevel)
    return eq


def MapQpToQlevel(pps, dsc_const, cpnt, qp):
    """
    :param pps: is_native_420, is_dsc_version_minor
    :param dsc_const: cpntBitDepth[0, 1], quantTableLuma[qp], quantTableChroma[qp]
    :return: qlevel
    """
    qlevel = 0

    isluma = (cpnt == 0 or cpnt == 3)
    isluma = isluma or ((pps.is_native_420) and (cpnt == 1))

    isYUV = (pps.is_dsc_version_minor == 2) and (dsc_const.cpntBitDepth[0] == dsc_const.cpntBitDepth[1])

    if isluma : qlevel = dsc_const.quantTableLuma[qp]
    else :
        # QP adjustment for YCbCr mode, Default : YCgCo
        if isYUV : qlevel = max(0, qlevel - 1)
        else :     qlevel = dsc_const.quantTableChroma[qp]

    return qlevel


def SamplePredict(defines, cpnt, hPos, sampModCnt, prevLine, currLine, predType, groupQuantizedResidual,
                  qLevel, cpntBitDepth ):

    # TODO h_offset_array_idx is equal to group count value
    # hPos = (0,1,2 -> 0) (3,4,5 -> 3) (6,7,8 -> 6) (9,10,11 -> 9)
    h_offset_array_idx = (hPos / defines.SAMPLES_PER_UNIT) * defines.SAMPLES_PER_UNIT + defines.PADDING_LEFT

    # organize samples into variable array defined in dsc spec
    c = prevLine[h_offset_array_idx-1]
    b = prevLine[h_offset_array_idx]
    d = prevLine[h_offset_array_idx+1]
    e = prevLine[h_offset_array_idx+2]
    a = currLine[h_offset_array_idx-1]

    filt_c = FILT3(prevLine[h_offset_array_idx-2], prevLine[h_offset_array_idx-1], prevLine[h_offset_array_idx])
    filt_b = FILT3(prevLine[h_offset_array_idx-1], prevLine[h_offset_array_idx], prevLine[h_offset_array_idx+1])
    filt_d = FILT3(prevLine[h_offset_array_idx], prevLine[h_offset_array_idx+1], prevLine[h_offset_array_idx+2])
    filt_e = FILT3(prevLine[h_offset_array_idx+1], prevLine[h_offset_array_idx+2], prevLine[h_offset_array_idx+3])

    if (predType == defines.PT_LEFT) : # Only at first line
        p = a
        if (sampModCnt == 1) :
            p = CLAMP(a + (groupQuantizedResidual[0] * QuantDivisor(qLevel)), 0, (1 << cpntBitDepth[cpnt]) - 1)
        elif (sampModCnt == 2) :
            p = CLAMP(a + (groupQuantizedResidual[0] + groupQuantizedResidual[1]) * QuantDivisor(qLevel),
                0, (1 << cpntBitDepth[cpnt]) - 1)

    elif (predType == defines.PT_MAP) : # MMAP
        diff = CLAMP(filt_c - c, -(QuantDivisor(qLevel) /2), QuantDivisor(qLevel) /2)
        if (hPos < defines.SAMPLES_PER_UNIT): blend_c = a
        else : blend_c = c + diff
        diff = CLAMP(filt_b - b, -(QuantDivisor(qLevel) / 2), QuantDivisor(qLevel) / 2)
        blend_b = b + diff
        diff = CLAMP(filt_d - d, -(QuantDivisor(qLevel) / 2), QuantDivisor(qLevel) / 2)
        blend_d = d + diff
        diff = CLAMP(filt_e - e, -(QuantDivisor(qLevel) / 2), QuantDivisor(qLevel) / 2)
        blend_e = e + diff

        if (sampModCnt == 0):
            p = CLAMP(a + blend_b - blend_c, min(a, blend_b), max(a, blend_b))
        elif (sampModCnt == 1):
            p = CLAMP(a + blend_d - blend_c + (groupQuantizedResidual[0] * QuantDivisor(qLevel)),
                      min(min(a, blend_b), blend_d),
                      max(max(a, blend_b), blend_d))
        else :
            p = CLAMP(a + blend_e - blend_c + (groupQuantizedResidual[0] + groupQuantizedResidual[1]) * QuantDivisor(qLevel),
                      min(min(a, blend_b), min(blend_d, blend_e)),
                      max(max(a, blend_b), max(blend_d, blend_e)))

    else : # Block prediction
        bp_offset = predType - defines.PT_BLOCK
        p = currLine[max(hPos + defines.PADDING_LEFT - 1 - bp_offset, 0)]

    return p


# output : pred_var
def PredictionLoop(pred_var, pps, dsc_const, defines, origLine, currLine, prevLine, hPos, vPos, sampModCnt, qp):
    """
    This function iterates for each unit (Y in first unit, Co in second unit, ...)
    :param pred_var: Main output of this function
    :param pps:
    :param dsc_const: constant values
    :param defines:
    :param origLine: Reconstructed pixel will be stored
    :param currLine:
    :param prevLine:
    :param hPos:
    :param vPos:
    :param sampModCnt:
    :param qp:
    :return:
    """
    # Loop for each unit (YYY CoCoCo CgCgCg)
    for unit in range(dsc_const.unitsPerGroup) :
        cpnt = dsc_const.unitCtype[unit]

        qlevel = MapQpToQlevel(pps, dsc_const, cpnt, qp)

        if (vPos == 0) :
            pred2use = defines.PT_LEFT # PT_LEFT is selected only at first line
        else :
            #### TODO modify it to be small variable
            pred2use = pred_var.prevLinePred[sampModCnt]

        if (pps.native_420) :
            ####### TODO native_420 mode
            raise NotImplementedError
        else :
            pred_x = SamplePredict(defines, cpnt, hPos, prevLine, currLine, pred2use, sampModCnt,
                                   pred_var.quantizedResidual[unit], qp, dsc_const.cpntBitDepth)

        actual_x = origLine[cpnt][hPos+defines.PADDING_LEFT]

        err_raw = actual_x - pred_x # get Quantized Residual
        err_raw_q = QuantizeResidual(err_raw, qlevel) # quantized residual check

        pred_mid = FindMidpoint(dsc_const, cpnt, qlevel, currLine[cpnt][min(pps.sliceWidth-1, hPos) + defines.PADDING_LEFT])
        err_mid = actual_x - pred_mid
        err_mid_q = QuantizeResidual(err_mid, qlevel) # MPP quantized residual check

        max_residual_size = MaxResidualSize(pps, dsc_const, cpnt, qp)

        # Midpoint residuals need to be bounded to BPC-QP in size, this is for some corner cases:
        # If an MPP residual exceeds this size, the residual is changed to the nearest residual with a size of cpntBitDepth - qLevel.
        # FIND NEAREST Q_RESIDUAL (6.4.5)
        if err_mid_q > 0:
            while(FindResidualSize(err_mid_q) > max_residual_size):
                err_mid_q -= 1
        else:
            while (FindResidualSize(err_mid_q) > max_residual_size):
                err_mid_q += 1

        ######### Save quantizedResidual #######
        pred_var.quantizedResidual[unit][sampModCnt] = err_raw_q
        pred_var.quantizedResidualMid[unit][sampModCnt] = err_mid_q

        #############################################################################
        ################  Inverse Quantization and Reconstruction (6.4.6) ###########

        #### Reconstruct prediction value
        maxval = (1 << dsc_const.cpntBitDepth[cpnt]) - 1
        recon_x = CLAMP(pred_x + (err_raw_q << qlevel), 0, maxval)

        if(dsc_const.full_ich_err_precision):
            absErr = abs(actual_x - recon_x)
        else :
            absErr = abs(actual_x - recon_x) >> (pps.bits_per_component - 8)
        ######### Save pred recon error #######
        pred_var.maxError[unit] = max(pred_var.maxError[unit], absErr)

        #### Reconstruct midpoint value
        recon_mid = pred_mid + (pred_var.quantizedResidualMid[unit][sampModCnt] << qlevel)
        recon_mid = CLAMP(recon_mid, 0, maxval)
        pred_var.midpointRecon[unit][sampModCnt] = recon_mid

        if(dsc_const.full_ich_err_precision):
            absErr = abs(actual_x - recon_mid)
        else :
            absErr = abs(actual_x - recon_mid) >> (pps.bits_per_component - 8)
        ######### Save mid recon error #######
        pred_var.maxMidError[unit] = max(pred_var.maxMidError[unit], absErr)

        #######################################################################
        #############################  Final output ###########################
        currLine[cpnt][hPos + defines.PADDING_LEFT] = recon_x
