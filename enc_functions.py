import os
import numpy as np
from dsc_utils import *
from init_enc_params import initDefines, initFlatVariables, initDscConstants, initIchVariables, initPredVariables, \
    initRcVariables, initVlcVariables

PRINT_DEBUG_OPT = False
PRINT_FUNC_CALL_OPT = False


def currline_to_pic(op, vPos, pps, dsc_const, defines, pic_val, currLine):
    if PRINT_FUNC_CALL_OPT: print("currline_to_pic has called!!")
    xs = pic_val.xs
    ## need to modify each axis
    # currLine_t = (currLine[0 : dsc_const.numComponents, 0 : defines.PADDING_LEFT]).transpose([1, 0])
    currLine_t = (currLine[0: dsc_const.numComponents, defines.PADDING_LEFT : ])
    currLine_t = currLine_t.transpose([1, 0])
    op[xs: xs + pps.slice_width, vPos, :] = currLine_t

    return op


def PopulateOrigLine(pps, dsc_const, hPos, vPos, pic):
    if PRINT_FUNC_CALL_OPT: print("PopulateOrigLine has called!!")
    width = dsc_const.sliceWidth

    return (pic[hPos : hPos + width, vPos, :]).transpose(1, 0)


def isFlatnessInfoSent(pps, rc_var):
    if PRINT_FUNC_CALL_OPT: print("isFlatnessInfoSent has called!!")
    is_flat_signaled = ((rc_var.masterQp >= pps.flatness_min_qp) and (rc_var.masterQp <= pps.flatness_max_qp))

    return is_flat_signaled


def isOrigFlatHIndex(hPos, origLine, rc_var, define, dsc_const, pps, flatQLevel):
    if PRINT_FUNC_CALL_OPT: print("isOrigFlatHIndex has called!!")
    fc1_start = 0
    fc1_end = 4
    fc2_start = 1
    fc2_end = 7
    t1_somewhat_flat = True
    t1_very_flat = True
    t2_somewhat_flat = True
    t2_very_flat = True
    flat_type_val = 0
    vf_thresh = pps.flatness_det_thresh

    qp = max(rc_var.masterQp - define.SOMEWHAT_FLAT_QP_DELTA, 0)
    thresh = np.copy(flatQLevel)

    is_end_of_slice = ((hPos + 1) >= dsc_const.sliceWidth)  # If group starts past the end of the slice, it can't be flat
    is_check_skip = ((hPos + 2) >= dsc_const.sliceWidth)  # Skip flatness check 2 if it only contains a single pixel

    if ((not (is_end_of_slice)) or (not (is_check_skip))):
        for cpnt in range(dsc_const.numComponents):
            max_val = -1
            min_val = 99999
            # print(currLine.shape, cpnt)

            for i in range(fc1_start, fc1_end):
                #if PRINT_DEBUG_OPT: print("CURRENT hPos is %d, i is %d" %(hPos, i))

                pixel_val = origLine[cpnt, define.PADDING_LEFT + hPos + i]

                if (max_val < pixel_val): max_val = pixel_val
                if (min_val > pixel_val): min_val = pixel_val

            is_somewhatflat_falied = (max_val - min_val) > max(vf_thresh, QuantDivisor(thresh[cpnt]))
            is_veryflat_failed = (max_val - min_val) > vf_thresh

            if (not is_somewhatflat_falied):
                t1_somewhat_flat = False

            if (not is_veryflat_failed):
                t1_very_flat = False

        # test2_condition = (not (t1_very_flat or t1_somewhat_flat))
        test2_condition = ((t1_somewhat_flat == False) and (t1_very_flat == False))

        # if (not test2_condition):
        #     a = 0

        # Left adjacent isn't flat, but current group & group to the right is flat
        #### Flat Test 2
        if (test2_condition):
            for cpnt in range(define.NUM_COMPONENTS):
                # vf_thresh = pps.flatness_det_thresh
                max_val = -1
                min_val = 99999

                for j in range(fc2_start, fc2_end):
                    # pixel_val = origLine[cpnt, define.PADDING_LEFT + hPos + j]
                    ## TODO : fix origline out of bound problem in flatness test 2...
                    ## Implemented temporarily pass keyword...
                    ##
                    try:
                        pixel_val = origLine[cpnt, define.PADDING_LEFT + hPos + j]

                    except:
                        pass

                    if (max_val < pixel_val): max_val = pixel_val
                    if (min_val > pixel_val): min_val = pixel_val

                is_somewhatflat_falied = ((max_val - min_val) > max(vf_thresh, QuantDivisor(thresh[cpnt])))
                is_veryflat_failed = ((max_val - min_val) > vf_thresh)

                if not is_somewhatflat_falied:
                    t2_somewhat_flat = False

                if not is_veryflat_failed:
                    t2_very_flat = False

    if (is_end_of_slice or is_check_skip):
        flat_type_val = 0
    elif (t1_very_flat):
        flat_type_val = 2
    elif (t1_somewhat_flat):
        flat_type_val = 1
    elif (t2_very_flat):
        flat_type_val = 2
    elif (t2_somewhat_flat):
        flat_type_val = 1
    else:
        flat_type_val = 0

    return flat_type_val


########### TODO hPos is not accurate
def FlatnessAdjustment(hPos, groupCount, pps, rc_var, flat_var, define, dsc_const, origLine, flatQLevel):
    if PRINT_FUNC_CALL_OPT: print("FlatnessAdjustment has called!!")
    pixelsInGroup = 3
    supergroup_cnt = (groupCount % define.GROUPS_PER_SUPERGROUP)
    flatness_index = hPos + (pixelsInGroup * define.GROUPS_PER_SUPERGROUP)

    if (supergroup_cnt == 0):
        flat_var.flatnessCnt = 0

    flat_var.flatnessMemory[flat_var.flatnessCnt] = isOrigFlatHIndex(flatness_index, origLine, rc_var, define, dsc_const,
                                                                    pps, flatQLevel)
    flat_var.flatnessCurPos = isOrigFlatHIndex(hPos, origLine, rc_var, define, dsc_const, pps, flatQLevel)
    flat_var.flatnessIdxMemory[flat_var.flatnessCnt] = supergroup_cnt

    if (flat_var.flatnessMemory[flat_var.flatnessCnt] > 0):  # If determined as flat
        flat_var.flatnessCnt += 1

    flat_var.IsQpWithinFlat = isFlatnessInfoSent(pps, rc_var)

    if (supergroup_cnt == 0):
        flat_var.firstFlat = flat_var.prevFirstFlat
        flat_var.flatnessType = flat_var.prevFlatnessType

    if ((supergroup_cnt == 3) and flat_var.IsQpWithinFlat):
        if (flat_var.firstFlat >= 0):
            flat_var.prevWasFlat = 1

        else:
            flat_var.prevWasFlat = 0

        flat_var.prevFirstFlat = -1
        if (flat_var.prevWasFlat):
            if ((flat_var.flatnessCnt >= 1) and (flat_var.flatnessMemort[0] > 0)):
                flat_var.prevFirstFlat = flat_var.flatnessIdxMemory[0]
                flat_var.prevFlatnessType = flat_var.flatnessMemory[0] - 1

        else:
            if (flat_var.flatnessCnt >= 1):
                flat_var.prevFirstFlat = flat_var.flatnessIdxMemory[0]
                flat_var.prevFlatnessType = flat_var.flatnessMemory[0] - 1

    elif ((supergroup_cnt == 3) and (not flat_var.IsQpWithinFlat)):
        flat_var.prevFirstFlat = -1

    flat_var.origIsFlat = 0
    if ((flat_var.firstFlat >= 0) and (supergroup_cnt == flat_var.firstFlat)):
        flat_var.origIsFlat = 1

    ## *MODEL NOTE* MN_FLAT_QP_ADJ
    if (hPos >= (pps.slice_width - 1)):
        flat_var.origIsFlat = 1
        flat_var.flatnessType = 1

    if (flat_var.origIsFlat and (rc_var.masterQp < pps.rc_range_parameters[define.NUM_BUF_RANGES - 1][1])):
        if ((flat_var.flatnessType == 0) or rc_var.masterQp < define.SOMEWHAT_FLAT_QP_THRESH):  # Somewhat flat
            rc_var.stQp = max(rc_var.stQp - define.SOMEWHAT_FLAT_QP_DELTA, 0)
            rc_var.prevQp = max(rc_var.prevQp - define.SOMEWHAT_FLAT_QP_DELTA, 0)

        else:  # very flat
            rc_var.stQp = define.VERY_FLAT_QP
            rc_var.prevQp = define.VERY_FLAT_QP


def CalcFullnessOffset(vPos, pixelCount, groupCnt, pps, define, dsc_const, vlc_var, rc_var):
    if PRINT_FUNC_CALL_OPT: print("CalcFullnessOffset has called!!")
    unity_scale = 1 << (define.RC_SCALE_BINARY_POINT)
    throttleFrac = 0  # from throttleFrac in dscstate structure

    temp_scaleAdjustCounter = rc_var.scaleAdjustCounter + 1
    flag1 = (groupCnt == 0)
    flag2 = ((vPos == 0) and (rc_var.currentScale > unity_scale))
    case_dec = (temp_scaleAdjustCounter > pps.scale_decrement_interval)
    case_inc = (temp_scaleAdjustCounter > pps.scale_increment_interval)
    flag3 = rc_var.scaleIncrementStart

    if (flag1):
        rc_var.currentScale = pps.initial_scale_value  # initial_scale_value = 32
        rc_var.scaleAdjustCounter = 1

    elif (flag2 and (not case_dec)):  # Reduce scale at beginning of slice
        rc_var.scaleAdjustCounter += 1

    elif (flag2 and case_dec):  # Reduce scale at beginning of slice
        rc_var.scaleAdjustCounter = 0
        rc_var.currentScale -= 1

    elif (flag3 and (not case_inc)):
        rc_var.scaleAdjustCounter += 1

    elif (flag3 and case_inc):
        rc_var.scaleAdjustCounter = 0
        rc_var.currentScale += 1

    flag0 = (vPos == 0)

    ## Account for first line boost
    ## Fixed values
    if (flag0):
        current_bpg_target = pps.first_line_bpg_ofs  # first_line_bpg_ofs =15
        increment = - (pps.first_line_bpg_ofs << define.OFFSET_FRACTIONAL_BITS)

    elif (not flag0):
        current_bpg_target = pps.nfl_bpg_offset  # nfl_bpg_offset = 288
        increment = pps.nfl_bpg_offset

    ## Account for 2nd line boost
    ## adds or substracts fixed values
    flag4 = rc_var.secondOffsetApplied
    flag5 = (vPos == 1)

    if (flag5 and (not flag4)):
        current_bpg_target += pps.second_line_bpg_offset
        increment += -(pps.second_line_bpg_offset << define.OFFSET_FRACTIONAL_BITS)
        rc_var.secondOffsetApplied = 1
        rc_var.rcXformOffset -= pps.second_line_offset_adj

    elif (flag5 and flag4):
        current_bpg_target += pps.second_line_bpg_offset
        increment += -(pps.second_line_bpg_offset << define.OFFSET_FRACTIONAL_BITS)

    elif ((not flag5)):
        current_bpg_target += (pps.nsl_bpg_offset >> define.OFFSET_FRACTIONAL_BITS)  # nsl_bpg_offset = 0
        increment += pps.nsl_bpg_offset

    # else:
    #     cond = pps.scale_increment_interval and (not rc_var.scaleIncrementStart) and (vPos > 0) and (rc_var.rcXformOffset > 0)
    #     if cond:
    #         rc_var.currentScale = 9
    #         rc_var.scaleAdjustCounter = 0
    #         rc_var.scaleIncrementStart = 1

    ## Account for initial delay boost
    num_pixels = 0
    pixelsInGroup = 3
    flag6 = (pixelCount < pps.initial_xmit_delay)
    flag7 = (pixelCount == 0)
    flag8 = (pps.scale_increment_interval and (not rc_var.scaleIncrementStart) and (vPos > 0) and (
            rc_var.rcXformOffset > 0))

    if (flag6 and flag7):
        num_pixels = pixelsInGroup
        num_pixels = min(pps.initial_xmit_delay - pixelCount, num_pixels)
        increment -= ((pps.bits_per_pixel) << (define.OFFSET_FRACTIONAL_BITS - 4))

    elif (flag6 and (not flag7)):
        num_pixels = pixelCount - rc_var.prevPixelCount
        num_pixels = min(pps.initial_xmit_delay - pixelCount, num_pixels)
        increment -= ((pps.bits_per_pixel * num_pixels) << (define.OFFSET_FRACTIONAL_BITS - 4))

    elif ((not flag6) and flag8):
        rc_var.currentScale = 9
        rc_var.scaleAdjustCounter = 0
        rc_var.scaleIncrementStart = 1

    rc_var.prevPixelCount = pixelCount
    current_bpg_target -= pps.slice_bpg_offset >> define.OFFSET_FRACTIONAL_BITS  # slice_bpg_offset = 68
    increment += pps.slice_bpg_offset  # slice_bpg_offset = 68
    throttleFrac += increment
    rc_var.rcXformOffset += throttleFrac
    throttleFrac = throttleFrac & ((1 << define.OFFSET_FRACTIONAL_BITS) - 1)

    if (rc_var.rcXformOffset < pps.final_offset):
        rc_var.rcOffsetClampEnable = 1

    if rc_var.rcOffsetClampEnable:
        rc_var.rcXformOffset = min(rc_var.rcXformOffset, pps.final_offset)

    return [rc_var.currentScale, rc_var.rcXformOffset]


def RateControl(vPos, pixelCount, sampModCnt, pps, dsc_const, ich_var, vlc_var, rc_var, flat_var, define):
    if PRINT_FUNC_CALL_OPT: print("RateControl has called!!")
    ## prev_fullness moved to main
    # prev_fullness = rc_var.bufferFullness
    mpsel = (vlc_var.midpointSelected).sum()
    rc_var.prevQp = rc_var.stQp
    rc_var.prev2Qp = rc_var.prevQp
    stQp = 0
    curQp = 0

    # pixelCount moved to enc_main
    # for i in range(sampModCnt):
    #     ### pixelCount???
    #     pass

    # Add up estimated bits for the Group, i.e. as if VLC sample size matched max sample size
    rcSizeGroupPrev = rc_var.rcSizeGroup
    rcSizeGroup = ((vlc_var.rcSizeUnit).sum()).item()

    # Set target number of bits per Group according to buffer fullness
    range_cfg = []

    ## Linear Transformation
    throttle_offset = rc_var.rcXformOffset
    throttle_offset -= pps.rc_model_size

    # *MODEL NOTE* MN_RC_XFORM
    rcBufferFullness = ((rc_var.currentScale * (rc_var.bufferFullness + rc_var.rcXformOffset)) >> (define.RC_SCALE_BINARY_POINT))

    overflowAvoid = (rc_var.bufferFullness + rc_var.rcXformOffset) > define.OVERFLOW_AVOID_THRESHOLD

    ### Pick the correct range
    # *MODEL NOTE* MN_RC_LONG_TERM
    thresh0 = pps.rc_buf_thresh[0] - pps.rc_model_size
    thresh1 = pps.rc_buf_thresh[1] - pps.rc_model_size
    thresh2 = pps.rc_buf_thresh[2] - pps.rc_model_size
    thresh3 = pps.rc_buf_thresh[3] - pps.rc_model_size
    thresh4 = pps.rc_buf_thresh[4] - pps.rc_model_size
    thresh5 = pps.rc_buf_thresh[5] - pps.rc_model_size
    thresh6 = pps.rc_buf_thresh[6] - pps.rc_model_size
    thresh7 = pps.rc_buf_thresh[7] - pps.rc_model_size
    thresh8 = pps.rc_buf_thresh[8] - pps.rc_model_size
    thresh9 = pps.rc_buf_thresh[9] - pps.rc_model_size
    thresh10 = pps.rc_buf_thresh[10] - pps.rc_model_size
    thresh11 = pps.rc_buf_thresh[11] - pps.rc_model_size
    thresh12 = pps.rc_buf_thresh[12] - pps.rc_model_size
    thresh13 = pps.rc_buf_thresh[13] - pps.rc_model_size

    range_cond14 = (rcBufferFullness > thresh13)
    range_cond13 = (thresh12 < rcBufferFullness <= thresh13)
    range_cond12 = (thresh11 < rcBufferFullness <= thresh12)
    range_cond11 = (thresh10 < rcBufferFullness <= thresh11)
    range_cond10 = (thresh9 < rcBufferFullness <= thresh10)
    range_cond9 = (thresh8 < rcBufferFullness <= thresh9)
    range_cond8 = (thresh7 < rcBufferFullness <= thresh8)
    range_cond7 = (thresh6 < rcBufferFullness <= thresh7)
    range_cond6 = (thresh5 < rcBufferFullness <= thresh6)
    range_cond5 = (thresh4 < rcBufferFullness <= thresh5)
    range_cond4 = (thresh3 < rcBufferFullness <= thresh4)
    range_cond3 = (thresh2 < rcBufferFullness <= thresh3)
    range_cond2 = (thresh1 < rcBufferFullness <= thresh2)
    range_cond1 = (thresh0 < rcBufferFullness <= thresh1)
    range_cond0 = (rcBufferFullness < thresh0)

    if range_cond0:
        j = 0
    elif range_cond1:
        j = 1
    elif range_cond2:
        j = 2
    elif range_cond3:
        j = 3
    elif range_cond4:
        j = 4
    elif range_cond5:
        j = 5
    elif range_cond6:
        j = 6
    elif range_cond7:
        j = 7
    elif range_cond8:
        j = 8
    elif range_cond9:
        j = 9
    elif range_cond10:
        j = 10
    elif range_cond11:
        j = 11
    elif range_cond12:
        j = 12
    elif range_cond13:
        j = 13
    elif range_cond14:
        j = 14

    if (rcBufferFullness > 0):
        raise ValueError("The RC model has overflowed.")

    # Add a group time of delay to RC calculation
    selected_range = rc_var.prevRange
    rc_var.prevRange = j

    bpg = (int(pps.bits_per_pixel * sampModCnt + 8) >> 4)  # Rounding fractional bits
    rcTgtBitGroup = max(0, bpg + (pps.rc_range_parameters[selected_range][2]).item() + rc_var.rcXformOffset)
    min_QP = (pps.rc_range_parameters[selected_range][0]).item()
    max_QP = (pps.rc_range_parameters[selected_range][1]).item()
    tgtMinusOffset = max(0, rcTgtBitGroup - pps.rc_tgt_offset_lo)
    tgtPlusOffset = max(0, rcTgtBitGroup + pps.rc_tgt_offset_hi)
    incr_amount = ((vlc_var.codedGroupSize - rcTgtBitGroup) >> 1)

    ### How about make this param canstant??
    ### SW
    predActivity = 0
    if (pps.native_420):
        predActivity = rc_var.prevQp + max(vlc_var.predictedSize[0].item(), vlc_var.predictedSize[1].item()) + vlc_var.predictedSize[2].item()

    elif (pps.native_422):
        predActivity = rc_var.prevQp + ((vlc_var.predictedSize.sum()) >> 1)

    else:  # 444 Mode
        predActivity = rc_var.prevQp + vlc_var.predictedSize[0].item() + max(vlc_var.predictedSize[1].item(), vlc_var.predictedSize[2].item())

    bitSaveThresh = dsc_const.cpntBitDepth[0].item() + dsc_const.cpntBitDepth[1].item() - 2

    ### *MODEL NOTE* MN_RC_SHORT_TERM
    ## bitSaveMode Decision Start...
    tmp_mppState = rc_var.mppState + 1
    bs_cond1 = ((vPos > 0) and (flat_var.firstFlat == -1))
    bs_cond2 = (tmp_mppState >= 2)
    bs_cond3 = ((not ich_var.ichSelected) and (mpsel >= 3))
    bs_cond4 = ((not ich_var.ichSelected) and (predActivity >= bitSaveThresh))
    bs_cond5 = ich_var.ichSelected

    bs_case1 = (bs_cond1 and bs_cond3 and bs_cond2)
    bs_case2 = (bs_cond1 and bs_cond3 and (not bs_cond2))
    bs_case3 = (bs_cond1 and bs_cond4)
    bs_case4 = (bs_cond1 and bs_cond5)
    bs_case5 = (bs_cond1 and (not bs_cond5))
    bs_case6 = (not bs_cond1)

    if bs_case1:
        rc_var.bitSaveMode = 2
        rc_var.mppState += 1

    elif bs_case2:
        rc_var.mppState += 1

    elif bs_case3:
        rc_var.bitSaveMode = rc_var.bitSaveMode

    elif bs_case4:
        rc_var.bitSaveMode = max(1, rc_var.bitSaveMode)

    elif bs_case5:
        rc_var.mppState = 0
        rc_var.bitSaveMode = 0

    elif bs_case6:
        rc_var.mppState = 0
        rc_var.bitSaveMode = 0

    ## Short-Term QP Adjustment Start...
    ## make condition to implement switch-case method
    ######### stqp Condition decision..######
    cond1 = overflowAvoid
    cond2 = (rc_var.bufferFullness <= 192)
    cond3 = (rc_var.bitSaveMode == 2)
    cond4 = (rc_var.bitSaveMode == 1)
    cond5 = (rc_var.rcSizeGroup == define.UNITS_PER_GROUP)
    cond6 = (rc_var.rcSizeGroup < tgtMinusOffset)
    # & (vlc_var.codedGroupSize < tgtMinusOffset))
    cond7 = ((rc_var.bufferFullness >= 64) and (vlc_var.codedGroupSize > tgtPlusOffset))
    cond8 = (not cond7)
    ##########################################

    if cond2:  # underflow Condition
        rc_var.stQp = min_QP  # cond2

    elif cond3:
        max_QP = min(pps.bits_per_component * 2 - 1, max_QP + 1)
        rc_var.stQp = rc_var.prevQp + 2

    elif cond4:
        max_QP = min(pps.bits_per_component * 2 - 1, max_QP + 1)
        rc_var.stQp = rc_var.prevQp

    elif cond5:
        min_QP = max(min_QP - 4, 0)
        rc_var.stQp = (rc_var.prevQp - 1)  # cond5

    elif cond6:
        rc_var.stQp = (rc_var.prevQp - 1)

    # avoid increasing QP immediately after edge
    elif cond7:  ## DO QP increment logic
        curQp = max(rc_var.prevQp, min_QP)

        inc_cond1 = (curQp == rc_var.prev2Qp)
        inc_cond2 = ((rc_var.rcSizeGroup * 2) < (rc_var.rcSizeGroupPrev * pps.rc_edge_factor))
        inc_cond3 = (rc_var.prev2Qp < curQp)
        inc_cond4 = (((rc_var.rcSizeGroup * 2) < (rc_var.rcSizeGroupPrev * pps.rc_edge_factor)) and
                     (curQp < pps.rc_quant_incr_limit0))
        inc_cond5 = (curQp < pps.rc_quant_incr_limit1)

        case1 = ((inc_cond1) and (inc_cond2))
        case2 = ((inc_cond1) and (not inc_cond2))
        case3 = ((not inc_cond1) and (inc_cond3) and (inc_cond4))
        case4 = ((not inc_cond1) and (inc_cond3) and (not inc_cond4))
        case5 = ((not inc_cond1) and (not inc_cond3) and (inc_cond5))
        case6 = ((not inc_cond1) and (not inc_cond3) and (not inc_cond5))

        if (case1 or case3 or case5): rc_var.stQp = (curQp + incr_amount)
        if (case2 or case4 or case6): rc_var.stQp = curQp

    elif cond8:
        rc_var.stQp = rc_var.prevQp

    elif cond1:  # overflow avoid condition
        rc_var.stQp = pps.rc_range_parameters[define.NUM_BUF_RANGES - 1, 0]  # cond1

    rc_var.stQp = CLAMP(rc_var.stQp, min_QP, max_QP)

    rc_var.rcSizeGroupPrev = rc_var.rcSizeGroup

    ## check rc buffer overflow
    is_overflowed = (rc_var.bufferFullness > pps.rcb_bits)

    if is_overflowed:
        raise ValueError("The buffer model has overflowed.")

    # masterQp update for next group
    rc_var.masterQp = rc_var.prevQp

    # print("RATE CONTROL FINISHED SUCCESSFULLY")

    # return rc_var.masterQp


def FindResidualSize(eq):
    if PRINT_FUNC_CALL_OPT: print("FindResidualSize has called!!")
    if (eq == 0):
        size_e = 0
    elif (-1 <= eq <= 0):
        size_e = 1
    elif (-2 <= eq <= 1):
        size_e = 2
    elif (-4 <= eq <= 3):
        size_e = 3
    elif (-8 <= eq <= 7):
        size_e = 4
    elif (-16 <= eq <= 15):
        size_e = 5
    elif (-32 <= eq <= 31):
        size_e = 6
    elif (-64 <= eq <= 63):
        size_e = 7
    elif (-128 <= eq <= 127):
        size_e = 8
    elif (-256 <= eq <= 255):
        size_e = 9
    elif (-512 <= eq <= 511):
        size_e = 10
    elif (-1024 <= eq <= 1023):
        size_e = 11
    elif (-2048 <= eq <= 2047):
        size_e = 12
    elif (-4096 <= eq <= 4095):
        size_e = 13
    elif (-8192 <= eq <= 8191):
        size_e = 14
    elif (-16384 <= eq <= 16383):
        size_e = 15
    elif (-32768 <= eq <= 32767):
        size_e = 16
    elif (-65536 <= eq <= 65535):
        size_e = 17
    elif (-131702 <= eq <= 131701):
        size_e = 18
    else:
        print("unexpectedly large residual size")
        raise ValueError
    return size_e


def MaxResidualSize(pps, dsc_const, cpnt, qp):
    ## this function has moved to mapQLevel var in enc_main loop.
    if PRINT_FUNC_CALL_OPT: print("MaxResidualSize has called!!")
    """
    :param pps: is_native_420, is_dsc_version_minor
    :param dsc_const: cpntBitDepth[cpnt], quantTableLuma[qp], quantTableChroma[qp]
    :return: qlevel
    """
    qlevel = MapQpToQlevel(pps, dsc_const, qp, cpnt)
    return dsc_const.cpntBitDepth[cpnt] - qlevel


def FindMidpoint(dsc_const, cpnt, qlevel, recon_value):
    if PRINT_FUNC_CALL_OPT: print("FindMidpoint has called!!")
    """
    :param cpntBitDepth[cpnt]:
    :param qlevel:
    :param recon_value
    :return:
    """
    recon_value = recon_value.astype(np.int32)
    midrange = 1 << dsc_const.cpntBitDepth[cpnt]
    midrange = midrange / 2
    #print(midrange)
    return int(midrange + (recon_value % (1 << qlevel)))


def QuantizeResidual(err, qlevel):
    if PRINT_FUNC_CALL_OPT: print("QuantizeResidual has called!!")
    """
    :param err:
    :param qlevel:
    :return:
    """
    # print(type(err))
    # print(err)
    # err = err.astype(np.int32)
    # print(type(err))
    # print(err)

    if err > 0:
        eq = (err + QuantOffset(qlevel)) >> qlevel
    else:
        eq = -1 * ((QuantOffset(qlevel) - err) >> qlevel)
    return eq


def MapQpToQlevel(pps, dsc_const, qp, cpnt):
    if PRINT_FUNC_CALL_OPT: print("MapQpToQlevel has called!!")
    """
    :param pps: is_native_420, is_dsc_version_minor
    :param dsc_const: cpntBitDepth[0, 1], quantTableLuma[qp], quantTableChroma[qp]
    :return: qlevel
    """
    qlevel = 0

    isluma = (cpnt == 0 or cpnt == 3)
    isluma = isluma or ((pps.native_420) and (cpnt == 1))

    isYUV = (pps.dsc_version_minor == 2) and (dsc_const.cpntBitDepth[0] == dsc_const.cpntBitDepth[1])

    if isluma:
        qlevel = dsc_const.quantTableLuma[qp]
    else:
        # QP adjustment for YCbCr mode, Default : YCgCo
        if isYUV:
            qlevel = max(0, qlevel - 1)
        else:
            qlevel = dsc_const.quantTableChroma[qp]

    return qlevel


def SamplePredict(defines, dsc_const, cpnt, hPos, prevLine, currLine, predType, sampModCnt, groupQuantizedResidual,
                  qLevel, cpntBitDepth):
    if PRINT_FUNC_CALL_OPT: print("SamplePredict has called!!")
    # TODO h_offset_array_idx is equal to group count value
    # hPos = (0,1,2 -> 0) (3,4,5 -> 3) (6,7,8 -> 6) (9,10,11 -> 9)
    h_offset_array_idx = int(hPos / defines.SAMPLES_PER_UNIT) * defines.SAMPLES_PER_UNIT + defines.PADDING_LEFT
    #if PRINT_DEBUG_OPT: print("[cpnt : %d] Current [h_offset_array_idxis %d], [hPos is %d]" %(cpnt, h_offset_array_idx, hPos))
    # organize samples into variable array defined in dsc spec
    c = prevLine[cpnt, h_offset_array_idx - 1].item()
    b = prevLine[cpnt, h_offset_array_idx].item()
    d = prevLine[cpnt, h_offset_array_idx + 1].item()
    e = prevLine[cpnt, h_offset_array_idx + 2].item()
    a = currLine[cpnt, h_offset_array_idx - 1].item()

    filt_c = FILT3(prevLine[cpnt, h_offset_array_idx - 2], prevLine[cpnt, h_offset_array_idx - 1], prevLine[cpnt, h_offset_array_idx])
    filt_b = FILT3(prevLine[cpnt, h_offset_array_idx - 1], prevLine[cpnt, h_offset_array_idx], prevLine[cpnt, h_offset_array_idx + 1])
    filt_d = FILT3(prevLine[cpnt, h_offset_array_idx], prevLine[cpnt, h_offset_array_idx + 1], prevLine[cpnt, h_offset_array_idx + 2])

    ## Exception : "filt_e" value is 0 when h_offset_array_idx is larger than (sliceWidth - 3)
    if (h_offset_array_idx == (dsc_const.sliceWidth - 2)):
        filt_e = 0

    else:
        filt_e = FILT3(prevLine[cpnt, h_offset_array_idx + 1], prevLine[cpnt, h_offset_array_idx + 2], prevLine[cpnt, h_offset_array_idx + 3])

    if (predType == defines.PT_LEFT):  # Only at first line
        p = a
        if (sampModCnt == 1):
            p = CLAMP(a + (groupQuantizedResidual[0] * QuantDivisor(qLevel)), 0, (1 << cpntBitDepth[cpnt]) - 1)
        elif (sampModCnt == 2):
            p = CLAMP(a + (groupQuantizedResidual[0] + groupQuantizedResidual[1]) * QuantDivisor(qLevel),
                      0, (1 << cpntBitDepth[cpnt]) - 1)

    elif (predType == defines.PT_MAP):  # MMAP
        diff = CLAMP(filt_c - c, -(QuantDivisor(qLevel) / 2), QuantDivisor(qLevel) / 2)

        if (hPos < defines.SAMPLES_PER_UNIT):
            blend_c = a

        else:
            blend_c = c + diff

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
        else:
            p = CLAMP(
                a + blend_e - blend_c + (groupQuantizedResidual[0] + groupQuantizedResidual[1]) * QuantDivisor(qLevel),
                min(min(a, blend_b), min(blend_d, blend_e)),
                max(max(a, blend_b), max(blend_d, blend_e)))

    else:  # Block prediction
        bp_offset = predType - defines.PT_BLOCK
        p = (currLine[cpnt, max(hPos + defines.PADDING_LEFT - 1 - bp_offset, 0)]).item()

    return p


# output : pred_var
def PredictionLoop(pred_var, pps, dsc_const, defines, origLine, currLine, prevLine, hPos, vPos, sampModCnt, mapQLevel,
                   maxResSize, qp):
    if PRINT_FUNC_CALL_OPT: print("PredictionLoop has called!!")
    """
    This function iterates for each unit (Y in first unit, Co in second unit, ...)
    :param pred_var: Main output of this function
    :param dsc_const: constant values
    :param origLine: Reconstructed pixel will be stored
    """
    # Loop for each unit (YYY CoCoCo CgCgCg)
    for unit in range(dsc_const.unitsPerGroup):
        cpnt = unit

        qlevel = mapQLevel[cpnt]

        if (vPos == 0):
            pred2use = defines.PT_LEFT  # PT_LEFT is selected only at first line

        else:
            #### TODO modify pred_var.prevLinePred[] to be short variable
            # pred2use = pred_var.prevLinePred[sampModCnt]
            pred2use = pred_var.prevLinePred[int(hPos/(defines.PRED_BLK_SIZE))].item()

        if (pps.native_420):
            ####### TODO native_420 mode
            raise NotImplementedError
        else:
            pred_x = SamplePredict(defines, dsc_const, cpnt, hPos, prevLine, currLine, pred2use, sampModCnt,
                                   pred_var.quantizedResidual[unit], qp, dsc_const.cpntBitDepth)

        ####### Calculate error for (blokc-prediction and midpoint-prediciton)
        actual_x = origLine[cpnt, hPos + defines.PADDING_LEFT].item()

        err_raw = int(actual_x - pred_x)  # get Quantized Residual
        err_raw_q = QuantizeResidual(err_raw, qlevel)  # quantized residual check

        pred_mid = FindMidpoint(dsc_const, cpnt, qlevel,
                                currLine[cpnt, min(dsc_const.sliceWidth - 1, hPos) + defines.PADDING_LEFT])
        err_mid = actual_x - pred_mid
        err_mid_q = QuantizeResidual(err_mid, qlevel)  # MPP quantized residual check

        err_raw_size = FindResidualSize(err_raw_q)
        err_mid_size = FindResidualSize(err_mid_q)

        # Midpoint residuals need to be bounded to BPC-QP in size, this is for some corner cases:
        # If an MPP residual exceeds this size, the residual is changed to the nearest residual with a size of cpntBitDepth - qLevel.
        # FIND NEAREST Q_RESIDUAL (6.4.5)
        max_residual_bit = maxResSize[cpnt]

        if (err_mid_size > max_residual_bit):
            if err_mid_q > 0:
                err_mid_q = 2 ** (max_residual_bit - 1) - 1
            else:
                err_mid_q = -1 * 2 ** (max_residual_bit - 1)

        ######### Save quantizedResidual #######
        pred_var.quantizedResidual[unit, sampModCnt] = err_raw_q
        pred_var.quantizedResidualMid[unit, sampModCnt] = err_mid_q

        if sampModCnt == 0:
            pred_var.max_size[unit] = err_raw_size
        else:
            pred_var.max_size[unit] = max(pred_var.max_size[unit], err_raw_size)
            if (pred_var.max_size[unit] >= maxResSize[unit]):
                pred_var.max_size[unit] = maxResSize[unit]

        pred_var.quantizedResidualSize[unit, sampModCnt] = err_raw_size
        ## TODO decoder part

        #############################################################################
        ################  Inverse Quantization and Reconstruction (6.4.6) ###########

        ############# Reconstruct prediction value ##############
        maxval = (1 << dsc_const.cpntBitDepth[cpnt]) - 1
        recon_x = int(CLAMP(pred_x + (err_raw_q << qlevel), 0, maxval))

        #### PIXEL VAL DEBUG....####
        print("[%d] [%d] cpnt : [%d] actual_x : [%d], recon_x : [%d]" %(hPos, vPos, cpnt, actual_x, recon_x))

        if (dsc_const.full_ich_err_precision):
            absErr = abs(actual_x - recon_x)
        else:
            absErr = abs(actual_x - recon_x) >> (pps.bits_per_component - 8)
        ######### Save pred recon error #######
        pred_var.maxError[unit] = max(pred_var.maxError[unit], absErr)

        ############# Reconstruct midpoint value  ##############
        #print(type(pred_var.quantizedResidualMid[unit][sampModCnt]))
        recon_mid = int(pred_mid + ((pred_var.quantizedResidualMid[unit, sampModCnt]).astype(np.int32) << qlevel))
        recon_mid = CLAMP(recon_mid, 0, maxval)

        if (dsc_const.full_ich_err_precision):
            absErr = abs(actual_x - recon_mid)

        else:
            absErr = abs(actual_x - recon_mid) >> (pps.bits_per_component - 8)
        ######### Save mid recon error #######
        pred_var.midpointRecon[unit, sampModCnt] = recon_mid
        pred_var.maxMidError[unit] = max(pred_var.maxMidError[unit], absErr)

        #######################################################################
        #############################  Final output ###########################
        currLine[cpnt, hPos + defines.PADDING_LEFT] = recon_x


## TODO check hPos and cpnt dependency (to parallelize computations)
## TODO can it be processed in every pixel-level? (Problem = predicted value is selected after VLC)
def BlockPredSearch(pred_var, pps, dsc_const, defines, currLine, cpnt, hPos):
    if PRINT_FUNC_CALL_OPT: print("BlockPredSearch has called!!")
    ################ Initial variables ###############
    min_bp_vector = 3
    max_bp_vector = 10
    pixel_mod_cnt = (hPos % defines.PRED_BLK_SIZE)
    cursamp = int(hPos / defines.PRED_BLK_SIZE) % defines.BP_SIZE
    ref_value = 1 << (dsc_const.cpntBitDepth[cpnt] - 1)
    max_cpnt = (dsc_const.numComponents - 1)
    bp_sads = (np.zeros(defines.BP_RANGE, )).astype(np.int32)

    if (pps.native_420):
        if (cpnt > 1):
            return ## BP Prohibited Condition...
        max_cpnt = 1

    ################ Reset variables ###############
    if (hPos == 0):
        pred_var.bpCount = 0  ## TODO new variable
        pred_var.lastEdgeCount = 10  # Arbitrary large value as initial condition  ## TODO new variable
        pred_var.lastErr[:, :, :] = 0
        # for i in range(dsc_const.numComponents):
        #     for j in range(defines.BP_SIZE):
        #         for candidate_vector in range(defines.BP_SIZE):
        #             # lastErr[NUM_COMPONENTS][BP_SIZE][BP_RANGE]
        #             # 3-pixel SAD's for each of the past 3 3-pixel-wide prediction blocks for each BP offset
        #             pred_var.lastErr[i][j][candidate_vector] = 0  ## TODO new variable

    if (pixel_mod_cnt == 0):
        for candidate_vector in range(defines.BP_RANGE):
            # predErr is summed over PRED_BLK_SIZE pixels
            pred_var.predErr[cpnt, candidate_vector] = 0

    ################ Does edge detected? detection process ###############
    ### TODO Executed every pixel-level
    recon_x = currLine[cpnt, hPos + defines.PADDING_LEFT].item()

    if (hPos == 0):
        # midpoint pixel value
        pixdiff = (recon_x - ref_value)

    else:
        # CurrentSample - LeftSample
        pixdiff = (recon_x - currLine[cpnt, hPos + defines.PADDING_LEFT - 1].item())

    pixdiff = abs(pixdiff)

    if (cpnt == 0):
        pred_var.edgeDetected = 0  ### Reset edgeDetected

    if (pixdiff > (defines.BP_EDGE_STRENGTH << (pps.bits_per_component - 8))): #Edge Occur Condition
        pred_var.edgeDetected = 1  ### Edge is detected

    if (cpnt == max_cpnt):  ### at the last component
        if pred_var.edgeDetected:
            pred_var.lastEdgeCount = 0

        else:
            pred_var.lastEdgeCount += 1  # edge is not detected at this pixel

    ################ Calculate difference between each component ###############
    ### TODO Executed every pixel-level
    # MAPED to... 0  1  2  3  4  5  6  7  8   9  10  11  12
    # THIS VALUE -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13
    for candidate_vector in range(defines.BP_RANGE):

        if (hPos > candidate_vector):
            # currLine[-1] ~ currLine[-13]
            pred_x = currLine[cpnt, max(hPos + defines.PADDING_LEFT - 1 - candidate_vector, 0)].item()
        else:
            pred_x = ref_value

        pixdiff = abs(recon_x - pred_x)
        modified_abs_diff = min(pixdiff >> (dsc_const.cpntBitDepth[cpnt].item() - 7), 0x3f)  # 6-bits
        # predErr is 8-bits
        pred_var.predErr[cpnt, candidate_vector] += modified_abs_diff

    ################ Select minimum SAD among [candidate_vector] ###############
    ### Last pixel in a group
    if (pixel_mod_cnt == (defines.PRED_BLK_SIZE - 1)):
        # Track last 3 3-pixel SADs for each component (each is 8 bit)

        for candidate_vector in range(defines.BP_RANGE):
            pred_var.lastErr[cpnt, cursamp, candidate_vector] = pred_var.predErr[cpnt, candidate_vector]

        if (cpnt == max_cpnt):
            for candidate_vector in range(defines.BP_RANGE):
                bp_sads[candidate_vector] = 0

                for i in range(defines.BP_SIZE): ## TODO CHECK (BP_RANGE -> BP_SIZE)
                    sad3x1 = 0

                    # Add up all components
                    for j in range(dsc_const.numComponents):
                        # (3 or 4) times of 8-bits
                        sad3x1 += pred_var.lastErr[j, i, candidate_vector] ## TODO CHECK INDEX AGAIN!! (i, j has been switched)

                    sad3x1 = min(511, sad3x1)  # sad3x1 is 9 bits

                    # Add up groups of BP_SIZE
                    bp_sads[candidate_vector] += sad3x1  # 11-bit SAD (3 times of 9-bits)
                # Each bp_sad can have a max value of 63*9 pixels * 3 components = 1701 or 11 bits

                bp_sads[candidate_vector] = ((bp_sads[candidate_vector].item()) >> 3)  # SAD is truncated to 8-bit for comparison

            min_err = bp_sads[0].item()
            min_pred = defines.PT_MAP

            # candidate_vector 3 ~ 9
            for candidate_vector in range((min_bp_vector - 1), max_bp_vector):
                # Ties favor smallest vector
                if (min_err > bp_sads[candidate_vector].item()):
                    min_err = bp_sads[candidate_vector].item()
                    min_pred = candidate_vector + defines.PT_BLOCK

            # Don't start algorithm until 10th pixel
            if ((pps.block_pred_enable) and (hPos >= 9)):
                if (min_pred > defines.PT_BLOCK):
                    pred_var.bpCount += 1

                else:
                    pred_var.bpCount = 0

            # BP is choosen in this condition
            if ((pred_var.bpCount >= 3) and (pred_var.lastEdgeCount < defines.BP_EDGE_COUNT)):
                pred_var.prevLinePred[int(hPos / defines.PRED_BLK_SIZE)] = min_pred

            else:
                pred_var.prevLinePred[int(hPos / defines.PRED_BLK_SIZE)] = defines.PT_MAP


def IsForceMpp(pps, dsc_const, rc_var):
    if PRINT_FUNC_CALL_OPT: print("IsForceMpp has called!!")
    maxBitsPerGroup = (dsc_const.pixelsInGroup * pps.bits_per_pixel + 15) >> 4
    adjFullness = rc_var.bufferFullness

    bugFixCondition = (pps.bits_per_pixel * pps.slice_width) & 0b1111
    tmp = rc_var.numBitsChunk + maxBitsPerGroup + 8

    force_mpp = 0
    if (bugFixCondition is not 0 and tmp == pps.chunk_size * 8) or (tmp > pps.chunk_size * 8):
        adjFullness -= 8
        if (adjFullness < maxBitsPerGroup - dsc_const.unitsPerGroup):
            force_mpp = 1
    ## Todo when VBR enabled

    return force_mpp


def VLCGroup(pps, defines, dsc_const, pred_var, ich_var, rc_var, vlc_var, flat_var, buf, groupCnt,
             FIFOs, seSizeFIFOs, Shifters, mapQLevel, maxResSize, adj_predicted_size):
    if PRINT_FUNC_CALL_OPT: print("VLCGroup has called!!")
    ######################### Declare variables #########################
    start_fullness = np.zeros(dsc_const.numSsps, ).astype(np.int32)
    max_size = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    max_err_p_mode = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    req_size = np.zeros((defines.MAX_UNITS_PER_GROUP, defines.SAMPLES_PER_UNIT)).astype(np.int32)
    prefix_size = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    suffix_size = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    add_prefix_one = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)

    #########################  Set control varaibles #########################
    if ((pps.bits_per_pixel == 0) and (3 * mapQLevel[0] <= 3 - adj_predicted_size[0])):
        ich_disallow = 1  # No ICH allowed for special case

    else:
        ich_disallow = 0

    forceMpp = IsForceMpp(pps, dsc_const, rc_var)

    ich_var.prevIchSelected = ich_var.ichSelected

    #########################################################################
    ####### Calculate maximum bit-width required for each component #########
    # maxError is the largest error value among identical component
    for unit in range(dsc_const.unitsPerGroup):
        if forceMpp or (pred_var.max_size[unit] == maxResSize[unit]):  # Use MPP
            vlc_var.midpointSelected[unit] = 1  # MPP error
            max_size[unit] = maxResSize[unit]  # maximum bit-width
            max_err_p_mode[unit] = pred_var.maxMidError[unit]

            for i in range(defines.SAMPLES_PER_UNIT):
                req_size[unit, i] = maxResSize[unit]

        else:
            vlc_var.midpointSelected[unit] = 0
            max_size[unit] = pred_var.max_size[unit]
            max_err_p_mode[unit] = pred_var.maxError[unit]

            for i in range(defines.SAMPLES_PER_UNIT):
                req_size[unit, i] = pred_var.quantizedResidualSize[unit, i]

    #########################################################################
    ############# Determines prefix and suffix size for P-mode ##############
    for unit in range(dsc_const.unitsPerGroup):
        enc_pred_size = max_size[unit] - adj_predicted_size[unit]
        add_prefix_one[unit] = 0

        ########## Predicted size is too small to hold max_size
        if adj_predicted_size[unit] < max_size[unit]:
            suffix_size[unit] = max_size[unit]

            if unit == 0:

                if (ich_var.prevIchSelected):
                    # ICH -> P (add '1' bit)
                    if (max_size[unit] == maxResSize[unit]):
                        prefix_size[unit] = enc_pred_size + 1

                    else:
                        prefix_size[unit] = enc_pred_size + 2  # smaller than Max bits (add '1' bit)
                else:
                    # P -> P (add '1' bit)
                    prefix_size[unit] = enc_pred_size + 1
            else:

                if max_size[unit] == maxResSize[unit]:
                    prefix_size[unit] = enc_pred_size

                else:
                    prefix_size[unit] = enc_pred_size + 1  # smaller than Max bits (add '1' bit)

            if unit == 0:

                if ich_var.prevIchSelected:

                    if (max_size[unit] != maxResSize[unit]):
                        add_prefix_one[unit] = 1
                else:
                    add_prefix_one[unit] = 1
            else:
                if max_size[unit] != maxResSize[unit]:
                    add_prefix_one[unit] = 1
        ######### Predicted size is sufficient to hold max_size
        else:
            suffix_size[unit] = adj_predicted_size[unit]

            if unit == 0:
                if ich_var.prevIchSelected:
                    prefix_size[unit] = 2

                else:
                    prefix_size[unit] = 1
            else:
                prefix_size[unit] = 1
            add_prefix_one[unit] = 1

    bits_p_mode = prefix_size[0] + prefix_size[1] + prefix_size[2] + prefix_size[3]
    bits_p_mode += defines.SAMPLES_PER_UNIT * (suffix_size[0] + suffix_size[1] + suffix_size[2] + suffix_size[3])

    #########################################################################
    ###################### Determines P or ICH mode ##########################
    if ich_var.prevIchSelected:
        ich_pfx = 1

    else:  # For escape code, no need to send trailing one for prefix
        ich_pfx = maxResSize[0] + 1 - adj_predicted_size[0]

    bits_ich_mode = ich_pfx + defines.ICH_BITS * dsc_const.pixelsInGroup  # length of encoded bits in case of ich mode

    sel_ich = IchDecision(pps, defines, flat_var, dsc_const, ich_var, ich_pfx, max_err_p_mode, bits_p_mode,
                          bits_ich_mode)

    if (sel_ich and ich_var.origWithinQerr and (not forceMpp) and (not ich_disallow)):  # At first unit
        ich_var.ichSelected = 1
        encoding_bits = bits_ich_mode  # encoded bit size for this group

    else:
        ich_var.ichSelected = 0
        encoding_bits = bits_p_mode

    if (groupCnt % defines.GROUPS_PER_SUPERGROUP) == 3 and flat_var.IsQpWithinFlat:
        encoding_bits += 1

    if (groupCnt % defines.GROUPS_PER_SUPERGROUP) == 0 and flat_var.firstFlat >= 0:
        if rc_var.masterQp >= defines.SOMEWHAT_FLAT_QP_THRESH:
            encoding_bits += 3

        else:
            encoding_bits += 2

    #########################################################################
    ########################### Encoding Process ############################
    # get prefix and encode each units
    ## Todo AddBits function
    VLCunit(dsc_const, vlc_var, flat_var, rc_var, ich_var, pred_var, defines, 0, groupCnt, add_prefix_one[0],
            max_size[0], prefix_size[0], suffix_size[0], maxResSize[0], FIFOs[0], ich_pfx)
    VLCunit(dsc_const, vlc_var, flat_var, rc_var, ich_var, pred_var, defines, 1, groupCnt, add_prefix_one[1],
            max_size[1], prefix_size[1], suffix_size[1], maxResSize[1], FIFOs[1], ich_pfx)
    VLCunit(dsc_const, vlc_var, flat_var, rc_var, ich_var, pred_var, defines, 2, groupCnt, add_prefix_one[2],
            max_size[2], prefix_size[2], suffix_size[2], maxResSize[2], FIFOs[2], ich_pfx)
    VLCunit(dsc_const, vlc_var, flat_var, rc_var, ich_var, pred_var, defines, 3, groupCnt, add_prefix_one[3],
            max_size[3], prefix_size[3], suffix_size[3], maxResSize[3], FIFOs[3], ich_pfx)

    if ich_var.ichSelected:
        encoding_bits = bits_ich_mode
        prefix_size[0] = ich_pfx
        prefix_size[1] = ich_pfx
        prefix_size[2] = ich_pfx
        prefix_size[3] = ich_pfx

        suffix_size[0] = defines.ICH_BITS
        suffix_size[1] = defines.ICH_BITS
        suffix_size[2] = defines.ICH_BITS
        suffix_size[3] = defines.ICH_BITS
        vlc_var.rcSizeUnit[0] = dsc_const.pixelsInGroup * defines.ICH_BITS + 1
        vlc_var.rcSizeUnit[1] = 0
        vlc_var.rcSizeUnit[2] = 0
        vlc_var.rcSizeUnit[3] = 0

    else:
        encoding_bits = bits_p_mode
        suffix_size[0] *= defines.SAMPLES_PER_UNIT
        suffix_size[1] *= defines.SAMPLES_PER_UNIT
        suffix_size[2] *= defines.SAMPLES_PER_UNIT
        suffix_size[3] *= defines.SAMPLES_PER_UNIT
        # rate control size uses max required size plus 1 (for prefix value of 0)
        vlc_var.rcSizeUnit[0] = max_size[0] * defines.SAMPLES_PER_UNIT + 1
        vlc_var.rcSizeUnit[1] = max_size[1] * defines.SAMPLES_PER_UNIT + 1
        vlc_var.rcSizeUnit[2] = max_size[2] * defines.SAMPLES_PER_UNIT + 1
        vlc_var.rcSizeUnit[3] = max_size[3] * defines.SAMPLES_PER_UNIT + 1

        # Predict size for next unit for this component ((required_size[0]+required_size[1]+2*required_size[2])/4)
        vlc_var.predictedSize[0] = (2 + req_size[0, 0] + req_size[0, 1] + 2 * req_size[0, 2]) >> 2
        vlc_var.predictedSize[1] = (2 + req_size[1, 0] + req_size[1, 1] + 2 * req_size[1, 2]) >> 2
        vlc_var.predictedSize[2] = (2 + req_size[2, 0] + req_size[2, 1] + 2 * req_size[2, 2]) >> 2
        vlc_var.predictedSize[3] = (2 + req_size[3, 0] + req_size[3, 1] + 2 * req_size[3, 2]) >> 2

    if ((groupCnt % defines.GROUPS_PER_SUPERGROUP == 3) and (flat_var.IsQpWithinFlat)):
        prefix_size[0] += 1
        encoding_bits += 1

    if ((groupCnt % defines.GROUPS_PER_SUPERGROUP == 0) and (flat_var.firstFlat >= 0)):
        if (rc_var.masterQp >= defines.SOMEWHAT_FLAT_QP_THRESH):
            prefix_size[0] += 3
            encoding_bits += 3

        else:
            prefix_size[0] += 2
            encoding_bits += 2

    for i in range(dsc_const.numSsps):
        seSizeFIFOs[i].fifo_put_bits(prefix_size.item(i) + suffix_size.item(i), 8)
        #fifo_put_bits(seSizeFIFOs[i], prefix_size[i] + suffix_size[i])

        if (FIFOs[i].fullness - start_fullness[i] > dsc_const.maxSeSize[i]):
        # if (prefix_size[i] + suffix_size[i] > dsc_const.maxSeSize[i]):
            #print("SE Size FIFO too small")
            pass

    if (groupCnt > (pps.muxWordSize + defines.MAX_SE_SIZE - 3)):
        ProcessGroupEnc(pps, dsc_const, vlc_var, buf, FIFOs, seSizeFIFOs, Shifters)

    vlc_var.codedGroupSize = encoding_bits


def RemoveBitsEncoderBuffer(pps, rc_var, dsc_const):
    if PRINT_FUNC_CALL_OPT: print("RemoveBitsEncoderBuffer has called!!")
    ## Function to remove one pixel's worth of bits from the encoder buffer model
    ## 6.8.1 section
    ## "chunkPixelTimes" is hPos + dsc_cfg->initial_xmit_delay (=512)
    ## "numBitsChunk" increases at 8 every pixel
    ##

    rc_var.bpgFracAccum += (pps.bits_per_pixel & 0xf)
    rc_var.bufferFullness -= (pps.bits_per_pixel >> 4) + (rc_var.bpgFracAccum >> 4)
    rc_var.numBitsChunk += (pps.bits_per_pixel >> 4) + (rc_var.bpgFracAccum >> 4)
    rc_var.bpgFracAccum = rc_var.bpgFracAccum & 0xf
    rc_var.chunkPixelTimes += 1

    if (rc_var.chunkPixelTimes >= dsc_const.sliceWidth):
        adjustment_bits = pps.chunk_size * 8 - rc_var.numBitsChunk
        rc_var.bufferFullness -= adjustment_bits

    rc_var.bpgFracAccum = 0
    rc_var.numBitsChunk = 0
    rc_var.chunkCount += 1
    rc_var.chunkPixelTimes = 0


def ProcessGroupEnc(pps, dsc_const, vlc_var, buf, FIFOs, seSizeFIFOs, Shifters):
    if PRINT_FUNC_CALL_OPT: print("ProcessGroupEnc has called!!")
    for i in range(pps.numSsps):

        if (Shifters[i].fullness < dsc_const.maxSeSize[i]):

            for j in range(int(pps.muxWordSize / 8)):
                sz = FIFOs[i].fullness

                if (sz >= 8):
                    #d = fifo_get_bits(FIFOs[i], 8, 0)
                    d = FIFOs[i].fifo_get_bits(8, 0)

                elif (0 < sz < 8):
                    #d = fifo_get_bits(FIFOs[i], sz, 0) << (8 - sz)
                    d = (int(FIFOs[i].fifo_get_bits(sz, 0)) << int(8 - sz))

                else:
                    d = 0

                ## put "d" into "buf" with size of "8 bits"
                ## byte count value of "buf" is stored in "postMuxNumBits"
                putbits(d, 8, buf)

                Shifters[i].fifo_put_bits(d, 8)
                ## fifo_put_bits(&dsc_state->shifter[i], d, 8); // put 'd' of '8-bits' into 'shifter'
                ## HAS MODIFIED TO BELOW EXPRESSION...
                ## SOLVED REMOVE THIS TYPE ERROR!!
                # if (isinstance(d, int)): ## Check "d" is a python integer or numpy integer...
                #     ##### Print out encoded data #####
                #     # 'buf_var' instantiated in dsc_main contains (outbuf, postMuxNumBits)
                #     # putbits(d, 8, buf_var)
                #     Shifters[i].fifo_put_bits(d, 8)
                #
                # else:
                #     d = d.item()
                #     Shifters[i].fifo_put_bits(d, 8)

        #sz = fifo_get_bits(seSizeFIFOs[i], 8, 0)
        sz = seSizeFIFOs[i].fifo_get_bits(8, 0)
        Shifters[i].fifo_get_bits(sz, 0)


def VLCunit(dsc_const, vlc_var, flat_var, rc_var, ich_var, pred_var, defines, unit, groupCnt, add_prefix_one,
            max_size, prefix_size, suffix_size, maxResSize, FIFO, ich_pfx):
    if PRINT_FUNC_CALL_OPT: print("VLCunit has called!!")
    ################################ Insert flat flag ####################################
    if ((unit == 0) and ((groupCnt % defines.GROUPS_PER_SUPERGROUP) == 3) and (flat_var.IsQpWithinFlat)):

        if (flat_var.prevFirstFlat < 0):
            addbits(vlc_var, FIFO, 0, 1)

        else:
            addbits(vlc_var, FIFO, 1, 1)

    ################################ Insert flat type ####################################
    if ((unit == 0) and (groupCnt % defines.GROUPS_PER_SUPERGROUP == 0) and (flat_var.firstFlat >= 0)):

        if rc_var.masterQp >= defines.SOMEWHAT_FLAT_QP_THRESH:
            addbits(vlc_var, FIFO, flat_var.flatnessType, 1)
        addbits(vlc_var, FIFO, flat_var.firstFlat, 2)

    ################################ ICH mode ####################################
    if (ich_var.ichSelected):
        #### ICH (unit == 0, prefix + suffix)
        if (unit == 0): ## LUMA Unit

            if ich_var.prevIchSelected:
                addbits(vlc_var, FIFO, 1, ich_pfx) ##

            else:
                addbits(vlc_var, FIFO, 0, ich_pfx) ##

            for i in range(dsc_const.pixelsInGroup):

                if (dsc_const.ichIndexUnitMap[i] == unit):
                    addbits(vlc_var, FIFO, ich_var.ichLookup[i].item(), defines.ICH_BITS)
        #### ICH Lookup (unit > 0, suffix)
        else:

            for i in range(dsc_const.pixelsInGroup):

                if (dsc_const.ichIndexUnitMap[i] == unit):
                    addbits(vlc_var, FIFO, ich_var.ichLookup[i].item(), defines.ICH_BITS)

    else:

        if add_prefix_one:
            addbits(vlc_var, FIFO, 1, prefix_size)

        else:
            addbits(vlc_var, FIFO, 0, prefix_size)

        for i in range(defines.SAMPLES_PER_UNIT):

            if max_size == maxResSize:
                addbits(vlc_var, FIFO, pred_var.quantizedResidualMid.item(unit, i), suffix_size)

            else:
                addbits(vlc_var, FIFO, pred_var.quantizedResidual.item(unit, i), suffix_size)


def IchDecision(pps, defines, flat_var, dsc_const, ich_var, alt_pfx, max_err_p_mode, bits_p_mode, bits_ich_mode):
    if PRINT_FUNC_CALL_OPT: print("IchDecision has called!!")
    ############# Error for each mode  ############
    log_err_p_mode = 0
    log_err_ich_mode = 0
    for i in range(dsc_const.unitsPerGroup):

        log_err_p_mode += ceil_log2(max_err_p_mode[i])
        log_err_ich_mode += ceil_log2(ich_var.maxIchError[i])

        if i == 0 and pps.dsc_version_minor == 1:
            log_err_p_mode <<= 1
            log_err_ich_mode <<= 1

    p_mode_cost = bits_p_mode + defines.ICH_LAMBDA * log_err_p_mode
    ich_mode_cost = bits_ich_mode + defines.ICH_LAMBDA * log_err_ich_mode

    if pps.dsc_version_minor == 2:
        if flat_var.flatnessCurPos == 2:
            decision = (log_err_ich_mode <= log_err_p_mode) and (ich_mode_cost < p_mode_cost)

        else:
            decision = ich_mode_cost < p_mode_cost
    else:
        decision = (log_err_ich_mode <= log_err_p_mode) and (ich_mode_cost < p_mode_cost)

    return decision


def UseICHistory(defines, dsc_const, ich_var, hPos, currLine):
    if PRINT_FUNC_CALL_OPT: print("UseICHistory has called!!")
    if defines.ICH_BITS == 0:
        return

    mod_hPos = hPos - dsc_const.pixelsInGroup + 1
    p = np.zeros(defines.NUM_COMPONENTS, dtype = np.int32)

    for i in range(dsc_const.pixelsInGroup):
        p[0] = ich_var.ichPixels[i, 0]
        p[1] = ich_var.ichPixels[i, 1]
        p[2] = ich_var.ichPixels[i, 2]
        p[3] = ich_var.ichPixels[i, 3]

        for cpnt in range(dsc_const.numComponents):
            currLine[cpnt, mod_hPos + i + defines.PADDING_LEFT] = p[cpnt]


def UpdateMidPoint(pps, defines, dsc_const, pred_var, vlc_var, hPos, currLine):
    if PRINT_FUNC_CALL_OPT: print("UpdateMidPoint has called!!")
    mod_hPos = hPos - dsc_const.pixelsInGroup + 1

    for i in range(defines.SAMPLES_PER_UNIT):
        if ((mod_hPos + i) <= (pps.slice_width - 1)):

            for unit in range(dsc_const.unitsPerGroup):
                if (vlc_var.midpointSelected[unit]):
                    currLine[unit, mod_hPos + defines.PADDING_LEFT + i] = pred_var.midpointRecon[unit, i]


def HistoryLookup(ich_var, defines, pps, dsc_const, prevLine, entry, hPos, first_line_flag, is_odd_line):
    if PRINT_FUNC_CALL_OPT: print("HistoryLookup has called!!")
    reserved = defines.ICH_SIZE - defines.ICH_PIXELS_ABOVE
    read_pixel = np.zeros(defines.NUM_COMPONENTS, )
    prevline_index = entry - reserved

    ############# settle particular range of hPos into same hPos #############
    # CASE 1 : pixelsInGroup == 3, CASE 2 : pixelsInGroup == 4
    # CASE 1 ||| hpos==(0,1,2 -> 1) | (3,4,5 -> 4) | (6,7,8 -> 7) | (9,10,11 -> 10)
    # CASE 2 ||| hpos==(0,1,2,3 -> 2) | (4,5,6,7 -> 6) | (8,9,10,11 -> 10) | (12,13,14,15 -> 14)
    mod_hPos = int(hPos / dsc_const.pixelsInGroup) * dsc_const.pixelsInGroup + int(dsc_const.pixelsInGroup / 2)

    if (pps.native_420 or pps.native_422):
        mod_hPos = CLAMP(hPos, 2, pps.slice_width - 1 - 2)

    else:
        # 3 <= mod_hPos <= end - 3
        temp_val = int(defines.ICH_PIXELS_ABOVE / 2)
        mod_hPos = CLAMP(mod_hPos, int(defines.ICH_PIXELS_ABOVE / 2), pps.slice_width - 1 - int(defines.ICH_PIXELS_ABOVE / 2))

    ############# Read out "ICH pixel value" at "entry" #############
    if ((not first_line_flag) and (prevline_index >= 0)):
        ## TODO native_420 mode
        ## TODO native_420 mode
        idx = mod_hPos + prevline_index - int(defines.ICH_PIXELS_ABOVE / 2) + defines.PADDING_LEFT

        read_pixel[0] = prevLine[0, mod_hPos + prevline_index - int(defines.ICH_PIXELS_ABOVE / 2) + defines.PADDING_LEFT]
        read_pixel[1] = prevLine[1, mod_hPos + prevline_index - int(defines.ICH_PIXELS_ABOVE / 2) + defines.PADDING_LEFT]
        read_pixel[2] = prevLine[2, mod_hPos + prevline_index - int(defines.ICH_PIXELS_ABOVE / 2) + defines.PADDING_LEFT]
        #read_pixel[2] = prevLine[3, mod_hPos + prevline_index - int(defines.ICH_PIXELS_ABOVE / 2) + defines.PADDING_LEFT]
        ## no needed for native_444 mode

    else:
        read_pixel[0] = ich_var.pixels[0, entry]
        read_pixel[1] = ich_var.pixels[1, entry]
        read_pixel[2] = ich_var.pixels[2, entry]
        read_pixel[3] = ich_var.pixels[3, entry]

    return read_pixel


def IsErrorPassWithBestHistory(ich_var, defines, pps, dsc_const, hPos, vPos, sampModCnt, modMapQLevel, orig, prevLine):
    if PRINT_FUNC_CALL_OPT: print("IsErrorPassWithBestHistory has called!!")
    if (sampModCnt == 0):
        ich_var.origWithinQerr = 1  # Reset wit no error

    max_qerr = np.zeros(defines.NUM_COMPONENTS, ).astype(np.int32)

    lowest_sad = 2 ** 30
    first_line_flag = ((vPos == 0) or (pps.native_420 and vPos == 1))
    ich_var.ichLookup[sampModCnt] = 99

    ich_prevline_start = (defines.ICH_SIZE - defines.ICH_PIXELS_ABOVE)
    ich_prevline_end = defines.ICH_SIZE

    if ((hPos == 0) and (vPos == 0)):
        ich_var.origWithinQerr = 0  # First pixel in a slice is error

    else:
        if (((not pps.native_420) and (vPos > 0)) or ((pps.native_420) and (vPos > 1))):
            # UL/U/UR always valid for non-first-lines
            for i in range(ich_prevline_start, ich_prevline_end):
                ich_var.valid[i] = True

        max_qerr[0] = int(QuantDivisor(modMapQLevel[0]) / 2)
        max_qerr[1] = int(QuantDivisor(modMapQLevel[1]) / 2)
        max_qerr[2] = int(QuantDivisor(modMapQLevel[2]) / 2)
        max_qerr[3] = int(QuantDivisor(modMapQLevel[3]) / 2)

        hit = 0
        for j in range(defines.ICH_SIZE):
            if (ich_var.valid[j].item()):
                # Calculate 'weightedSad'
                # Let Find the Minimum 'weightedSad' Value
                weighted_sad = 0
                ich_pixel = HistoryLookup(ich_var, defines, pps, dsc_const, prevLine, j, hPos, first_line_flag,
                                          (vPos % 2))

                diff0 = abs(ich_pixel[0] - orig[0])
                diff1 = abs(ich_pixel[1] - orig[1])
                diff2 = abs(ich_pixel[2] - orig[2])
                diff3 = abs(ich_pixel[3] - orig[3])

                if (dsc_const.numComponents == 3):
                    if ((diff0 <= max_qerr[0]) and (diff1 <= max_qerr[1]) and (diff2 <= max_qerr[2])):
                        hit = 1

                if (dsc_const.numComponents == 4):
                    if ((diff0 <= max_qerr[0]) and (diff1 <= max_qerr[1]) and (diff2 <= max_qerr[2]) and (
                            diff3 <= max_qerr[3])):
                        hit = 1

                if (pps.native_422):
                    weighted_sad = 2 * diff0 + diff1 + diff2 + 2 * diff3

                elif (not pps.native_420) or (pps.dsc_version_minor == 1):
                    weighted_sad = 2 * diff0 + diff1 + diff2 ## YCoCg chroma has an extra bit

                else:
                    weighted_sad = diff0 + diff1 + diff2

                if (lowest_sad > weighted_sad): ## Find lowest SAD
                    lowest_sad = weighted_sad
                    ich_var.ichPixels[sampModCnt, 0] = ich_pixel[0]
                    ich_var.ichPixels[sampModCnt, 1] = ich_pixel[1]
                    ich_var.ichPixels[sampModCnt, 2] = ich_pixel[2]
                    ich_var.ichPixels[sampModCnt, 3] = ich_pixel[3]
                    ich_var.ichLookup[sampModCnt] = j

        # debugging
        if ((ich_var.ichLookup[sampModCnt] == 99) and (ich_var.valid[0].item())):
            print("ICH search failed : [weight_sad = %d]" %weighted_sad) ## TODO : NO ICH search Fail Case...

        if (hit):
            ##### Check the ICH error per components #####
            # Y -> Co -> Cg -> (Y2)
            for i in range(dsc_const.unitsPerGroup):
                if (dsc_const.full_ich_err_precision):
                    absErr = abs(ich_var.ichPixels[sampModCnt, i] - orig[i])

                else:
                    absErr = abs(ich_var.ichPixels[sampModCnt, i] - orig[i]) >> (pps.bits_per_component - 8)
                ich_var.maxIchError[i] = max(ich_var.maxIchError[i], absErr)
        else:
            ich_var.origWithinQerr = 0


def UpdateHistoryElement(pps, defines, dsc_const, ich_var, vlc_var, prevLine, hPos, vPos, recon):
    if PRINT_FUNC_CALL_OPT: print("UpdateHistoryElement has called!!")
    first_line_flag = ((vPos == 0) or (pps.native_420 and (vPos == 1)))
    read_pixel = np.array(4, dtype = np.uint32)
    # 32 or 25 (=32-7)
    if (first_line_flag):
        reserved = defines.ICH_SIZE
    else:
        reserved = defines.ICH_SIZE - defines.ICH_PIXELS_ABOVE
    # Update the ICH with recon as the MRU
    hit = 0
    loc = reserved - 1  # LRU bit (rightmost bit), if no match delete LRU

    for j in range(reserved):  # 'j' notifies the entry
        if (not (ich_var.valid[j].item())):
            loc = j
            break

        else:
            read_pixel = HistoryLookup(ich_var, defines, pps, dsc_const, prevLine, j, hPos, first_line_flag, vPos % 2)
            # pecific hPos within group is not critical since hits against UL/U/UR don't have specific detection
            hit = 1

            for cpnt in range(dsc_const.numComponents):
                if read_pixel[cpnt] != recon[cpnt]:
                    hit = 0
            # if ICH was selected to (encode or decode) for previous pixel
            # and all components of ICH[j] are same with those of recon[]
            if hit and ich_var.ichSelected:  ## TODO decoder part
                loc = j
                break  # Found one
    j = 0

    ## shifting ICH pixels
    for cpnt in range(dsc_const.numComponents):
        ## Delete from current position "loc" (or delete "LRU")
        for j in range(loc, 0, -1):
            #print("j : ", j)
            ich_var.pixels[cpnt, j] = ich_var.pixels[cpnt, j - 1]
        ich_var.valid[loc] = True

        # Insert the most recently reconstructed pixel into MRU
        ich_var.pixels[cpnt, 0] = recon[cpnt]
        ich_var.valid[0] = True


def SampToLineBuf(dsc_const, pps, cpnt, x):
    if PRINT_FUNC_CALL_OPT: print("SampToLineBuf has called!!")
    ## Line Storage (6.3)
    ## Allocate Storage to Store Reconstructed Pixel Value
    shift_amount = max(dsc_const.cpntBitDepth[cpnt] - pps.line_buf_depth, 0)
    storedSample = 0

    if (shift_amount > 0):
        rounding = 1 << (shift_amount - 1)

    else:
        rounding = 0

        # Max value = 2^line_buf_depth - 1
        storedSample = min((x + rounding) >> shift_amount, (1 << pps.line_buf_depth) - 1)

    return_val = storedSample << shift_amount
    return return_val
