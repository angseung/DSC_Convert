import numpy as np
from init_enc_params import *
from enc_functions import *
from dsc_fifo import DSCFifo
from dsc_enc_buf import *

def dsc_encoder(pps, pic, op, buf, pic_val):
    ################ Declare variables used to each block ################
    defines = initDefines(pps)
    dsc_const = initDscConstants(pps, defines)
    ich_var = initIchVariables(defines)
    pred_var = initPredVariables(defines, dsc_const)
    flat_var = initFlatVariables(defines)
    vlc_var = initVlcVariables(defines)
    rc_var = initRcVariables()

    ########### Declare buffers ########
    # Line Buffer Axis : [Component, hPos]
    currLine = np.zeros((defines.NUM_COMPONENTS, pps.chunk_size + defines.PADDING_LEFT)).astype(np.int32)
    tmp_prevLine = np.zeros((defines.NUM_COMPONENTS, pps.chunk_size + defines.PADDING_LEFT)).astype(np.int32)
    prevLine = np.zeros((defines.NUM_COMPONENTS, pps.chunk_size + defines.PADDING_LEFT)).astype(np.int32)
    origLine = np.zeros((defines.NUM_COMPONENTS, pps.chunk_size + defines.PADDING_LEFT)).astype(np.int32)

    ## LINE BUFFER VALUE INITIALIZE
    for cpnt in range(dsc_const.numComponents):
        initvalue = (1 << (dsc_const.cpntBitDepth[cpnt] - 1))
        currLine[cpnt, :] = initvalue
        tmp_prevLine[cpnt, :] = initvalue
        prevLine[cpnt, :] = initvalue

    #fifo_size = int(((pps.muxWordSize + defines.MAX_SE_SIZE - 1) * (defines.MAX_SE_SIZE + 7)) / 8) * 8
    fifo_size = int(((pps.muxWordSize + defines.MAX_SE_SIZE - 1) * (defines.MAX_SE_SIZE) + 7) / 8)
    seSizefifo_size = int((8 * (pps.muxWordSize + defines.MAX_SE_SIZE - 1) + 7) / 8)
    shifter_size = int((pps.muxWordSize + defines.MAX_SE_SIZE + 7) / 8)

    ## Declare FIFO and seSizeFIFO ##
    FIFO_Y = DSCFifo(fifo_size)
    FIFO_Co = DSCFifo(fifo_size)
    FIFO_Cg = DSCFifo(fifo_size)
    FIFO_Y2 = DSCFifo(fifo_size)
    FIFOs = [FIFO_Y, FIFO_Co, FIFO_Cg, FIFO_Y2]

    seSizeFifo_Y = DSCFifo(seSizefifo_size)
    seSizeFifo_Co = DSCFifo(seSizefifo_size)
    seSizeFifo_Cg = DSCFifo(seSizefifo_size)
    seSizeFifo_Y2 = DSCFifo(seSizefifo_size)
    seSizeFIFOs = [seSizeFifo_Y, seSizeFifo_Co, seSizeFifo_Cg, seSizeFifo_Y2]

    ## TODO REMOVE FIFO-Class SHIFET, then change this to Value!!
    Shifter_Y = DSCFifo(shifter_size)
    Shifter_Co = DSCFifo(shifter_size)
    Shifter_Cg = DSCFifo(shifter_size)
    Shifter_Y2 = DSCFifo(shifter_size)
    Shifters = [Shifter_Y, Shifter_Co, Shifter_Cg, Shifter_Y2]
    #############################################################

    oldQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    mapQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    modMapQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    flatQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    maxResSize = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
    adj_predicted_size = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)

    lbufWidth = pps.slice_width + defines.PADDING_LEFT + defines.PADDING_RIGHT
    hPos = 0
    vPos = 0
    sampModCnt = 0
    groupCnt = 0
    pixelCount = 0
    done = 0

    ###########################################################
    ######################## Main Loop ########################
    while (not done):
        # print("NOW PROCESSING [%04d][%04d]TH LINE IN A SCLICE..." %(hPos, vPos))
        #################### Get input line ###################
        if (hPos == 0):
            ## Get input image when the first pixel of each line starts
            origLine[0 : dsc_const.numComponents, :] = 0 ## Clear OrigLine Buffer...
            origLine[0 : dsc_const.numComponents, (defines.PADDING_LEFT) : ] = PopulateOrigLine(pps, dsc_const, hPos, vPos, pic)

        ## ICH DEBUG ONLY
        if (hPos == 749):
            a = 10

        ################ Initialization ###################
        ## TODO write below codes into each corresponding functions
        if (sampModCnt == 0):

            modified_qp = min(2*pps.bits_per_component - 1, rc_var.masterQp + 2)
            flat_qp = max(rc_var.masterQp - pps.somewhat_flat_qp_delta, 0)

            for i in range(defines.NUM_COMPONENTS):
                oldQLevel[i] = mapQLevel[i]
                mapQLevel[i] = MapQpToQlevel(pps, dsc_const, rc_var.masterQp, i)
                modMapQLevel[i] = MapQpToQlevel(pps, dsc_const, modified_qp, i)
                flatQLevel[i] = MapQpToQlevel(pps, dsc_const, flat_qp, i)
                maxResSize[i] = dsc_const.cpntBitDepth[i] - mapQLevel[i]

                pred_size = vlc_var.predictedSize[i] + (oldQLevel[i] - mapQLevel[i])
                adj_predicted_size[i] = CLAMP(pred_size, 0, maxResSize[i] - 1)

            ## Reset valid control of ICH
            ## Reset ICH at beginning of each line if multiple slices per line
            if (((hPos == 0) and (vPos == 0)) or # Beginning of slice
                    ((hPos == 0) and (not (pps.slice_width == pps.pic_width)))): # End of Slice!
                ich_var.valid[:] = 0 # zero setting

                ## DELETE UNNESESSARY LOOP
                # for idx in range(defines.ICH_SIZE):
                #     ich_var.valid[idx] = 0

        #################### Predict operation ###################

        # if (hPos == 1270):
        #     pass

        PredictionLoop(pred_var, pps, dsc_const, defines, origLine, currLine, prevLine, hPos, vPos, sampModCnt,
                                             mapQLevel, maxResSize, rc_var.masterQp)

        #################### P or ICH Selection ###################
        ###################### ICH operation ######################
        orig = np.zeros(defines.NUM_COMPONENTS, ).astype(np.int32)
        tmp = (origLine[0 : dsc_const.numComponents, hPos + defines.PADDING_LEFT])
            #.reshape((dsc_const.numComponents, 1))
        orig[0 : dsc_const.numComponents] = tmp
        IsErrorPassWithBestHistory(ich_var, defines, pps, dsc_const, hPos, vPos, sampModCnt, modMapQLevel, orig,
                                       prevLine)

        ## CHECK ich_var.origWithinQerr for Debug...

        #####################################################################
        ###################### Last pixel in a a group ######################
        #################### Flatness adjustment ###################
        if ((sampModCnt == (dsc_const.pixelsInGroup - 1)) or (hPos == (dsc_const.sliceWidth - 1))):
            ## Last Pixel in a Group "OR" Outbound pixel outside of slice
            #print("hPos is %d, campModCnt is %d" %(hPos, sampModCnt))
            FlatnessAdjustment(hPos, groupCnt, pps, rc_var, flat_var, defines, dsc_const, origLine, flatQLevel)

            if (sampModCnt < (dsc_const.pixelsInGroup - 1)):

                if (sampModCnt == 0):
                    ich_var.ichLookup[1] = ich_var.ichLookup[0]
                    ich_var.ichLookup[2] = ich_var.ichLookup[0]
                    hPos += 2 ## set hPos to indicate the last pixel in a group

                if (sampModCnt == 1):
                    ich_var.ichLookup[2] = ich_var.ichLookup[1]
                    hPos += 1 ## set hPos to indicate the last pixel in a group

            ######################### Variable Length Encoding (VLC) ####################
            VLCGroup(pps, defines, dsc_const, pred_var, ich_var, rc_var, vlc_var, flat_var, buf, pixelCount, groupCnt,
                     FIFOs, seSizeFIFOs, Shifters, mapQLevel, maxResSize, adj_predicted_size, vPos, hPos)

            # bufferFullness = 0 ## Declair bufferFullness Variable
            rc_var.bufferFullness += vlc_var.codedGroupSize # Increase buffer fullness
            bufferFullness = rc_var.bufferFullness ## 2020.07.30 Revision

            # print("[%d] [%d] codedGroupSize : [%d] Current Buffer Fullness : [%d]"
            #       %(vPos, hPos, vlc_var.codedGroupSize, rc_var.bufferFullness))

            if (bufferFullness > pps.rcb_bits):
                ## This check may actually belong after tgt_bpg has been subtracted
                print("The buffer model has overflowed.  This probably occurred due to an error in the rate control parameter programming.")
                print("ERROR: RCB overflow; size is %d, tried filling to %d", pps.rcb_bits, bufferFullness)
                exit(1)

            ########### The final reconstructed pixel value ############
            if (ich_var.ichSelected):
                UseICHistory(defines, dsc_const, ich_var, hPos, currLine)
                # print("ICH Selected in [%d] [%d]" %(vPos, hPos))

            else:
                UpdateMidPoint(pps, defines, dsc_const, pred_var, vlc_var, hPos, currLine)
                # print("MPP Selected in [%d] [%d]" % (vPos, hPos))

            ########### Update ICH pixels ############
            if ((not (defines.ICH_BITS == 0)) and (hPos < (dsc_const.sliceWidth - 1))): ## Skip the Update ICH of the last group
                mod_hPos = (hPos - 2)

                ich_p = np.zeros(defines.NUM_COMPONENTS, dtype = np.int32)

                for i in range(dsc_const.pixelsInGroup):
                    for cpnt in range(dsc_const.numComponents):
                        ich_p[cpnt] = currLine[cpnt, mod_hPos + i + defines.PADDING_LEFT]

                    UpdateHistoryElement(pps, defines, dsc_const, ich_var, vlc_var, prevLine, hPos, vPos, ich_p)

            ########### Predict MMAP vs BP for the next line ############
            for mod_hPos in range(hPos - 2, hPos + 1):
                for cpnt in range(dsc_const.numComponents):
                    BlockPredSearch(pred_var, pps, dsc_const, defines, currLine, cpnt, mod_hPos) ## Modified from hPos to mod_hPos

            ########### Store reconstructed value in prevLineBuf (double buffer) ############
            for mod_hPos in range(hPos - 2 + defines.PADDING_LEFT, hPos + 1 + defines.PADDING_LEFT):

                for cpnt in range(dsc_const.numComponents):

                    if ((pps.native_420) and (cpnt == dsc_const.numComponents - 1)):
                        tmp_prevLine[cpnt + (vPos % 2), mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt,
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + dsc_const.sliceWidth - 1)])

                    else:
                        tmp_prevLine[cpnt, mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt,
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + dsc_const.sliceWidth - 1)])

            ################################## Rate controller  #############################
            [rc_var.currentScale, rc_var.rcXformOffset] = CalcFullnessOffset(vPos, pixelCount,
                                                                               groupCnt, pps, defines, dsc_const, vlc_var, rc_var)
            groupCnt += 1 ## TODO groupCnt increase timing
            ############################### From RateControl Func...
            rc_var.prevFullness = rc_var.bufferFullness

            for i in range(dsc_const.pixelsInGroup):
                pixelCount += 1 ## pixelCount MUST BE INCREASED IN ENC_MAIN LOOP FOR A HARDWARE IMPLEMENTATION...

                if (pixelCount >= pps.initial_xmit_delay):
                    RemoveBitsEncoderBuffer(pps, rc_var, dsc_const)

            RateControl(hPos, vPos, pixelCount, sampModCnt, pps, dsc_const, ich_var, vlc_var, rc_var, flat_var, defines)
            # print("Currnt Position : [%d] [%d], masterQp is [%d]" %(vPos, hPos, rc_var.masterQp))
            ## masterQp decision is done in Rate Control function...
            # rc_var.masterQp = rc_var.prevQp

            ### RESET RESIDUAL VALUES...
            vlc_var.midpointSelected[:] = 0
            pred_var.quantizedResidual[:, :] = 0
            pred_var.quantizedResidualMid[:, :] = 0

            ### CLEAR ICH ERRORS...
            pred_var.maxError[:] = 0
            pred_var.maxMidError[:] = 0
            ich_var.maxIchError[:] = 0
            ################ END OF LAST GROUP PROCESSING

        ################## Counter controller ############################
        hPos += 1
        sampModCnt += 1

        ### RESET sampModCnt Value
        if (sampModCnt == 3):
            # groupCnt += 1 # increases to the end of slice /// MOVED TO UPPER LINE!!
            sampModCnt = 0

        ## End of a line
        if (hPos >= dsc_const.sliceWidth):

            # Mapping Reconstructed Value to Out Picture 'op'
            op = currline_to_pic(op, vPos, pps, dsc_const, defines, pic_val, currLine)

            # Fill tmp_prevLine outside of dsc_state->sliceWidth (PADDING_LEFT and PADDING_RIGHT)
            for mod_hPos in range(defines.PADDING_LEFT):
                for cpnt in range(dsc_const.numComponents):

                    if ((pps.native_420) and (cpnt == (dsc_const.numComponents - 1))):
                        tmp_prevLine[cpnt + (vPos % 2), mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt,
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])

                    else:
                        tmp_prevLine[cpnt, mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt,
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])

            # for PADDING RIGHT
            for mod_hPos in range(defines.PADDING_LEFT + dsc_const.sliceWidth, lbufWidth):
                for cpnt in range(dsc_const.numComponents):

                    if ((pps.native_420) and (cpnt == (dsc_const.numComponents - 1))):
                        tmp_prevLine[cpnt + (vPos % 2), mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt, CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])

                    else:
                        tmp_prevLine[cpnt, mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt, CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])

            # Deliver the value from "tmp_prevLine" to "prevLine"
            for cpnt in range(dsc_const.numComponents):
                for i in range(lbufWidth):

                    if ((pps.native_420) and (cpnt == (dsc_const.numComponents - 1))):
                        prevLine[cpnt + (vPos % 2), i] = tmp_prevLine[cpnt + (vPos % 2), i]

                    else:
                        prevLine[cpnt, i] = tmp_prevLine[cpnt, i]

            hPos = 0
            vPos += 1

            if (vPos >= pps.slice_height):
                done = 1

    ## while Done!

    if (not (sampModCnt == 0)): ## Pad last unit wih 0's if needed
        pred_var.quantizedResidualSize[:, :] = 0
        VLCGroup(pps, defines, dsc_const, pred_var, ich_var, rc_var, vlc_var, flat_var, buf, groupCnt,
                 FIFOs, seSizeFIFOs, Shifters, mapQLevel, maxResSize, adj_predicted_size)

    while (seSizeFIFOs[0].fullness > 0):
        ProcessGroupEnc(pps, dsc_const, vlc_var, buf, FIFOs, seSizeFIFOs, Shifters, vPos, hPos)
        # End of While (Encoding process)

    #######################################################################

    ## Erase FIFO data...
    for i in range(defines.MAX_NUM_SSPS):
        seSizeFIFOs[i].fifo_free()
        FIFOs[i].fifo_free()
        Shifters[i].fifo_free()






