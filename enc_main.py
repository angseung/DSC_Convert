import numpy as np
from init_enc_params import *
from enc_functions import *

def dsc_encoder(pps, pic, op, buf):
    ################ Declare variables used to each block ################
    defines = initDefines(pps)
    dsc_const = initDscConstants(pps, defines)
    ich_var = initIchVariables(defines)
    pred_var = initPredVariables(defines)
    flat_var = initFlatVariables(defines)
    vlc_var = initVlcVariables(defines)
    rc_var = initRcVariables()

    ########### Declare buffers ########
    # Line Buffer Axis : [Component, hPos]
    currLine = np.zeros((3, pps.chunk_size + defines.PADDING_LEFT))
    tmp_prevLine = np.zeros((3, pps.chunk_size + defines.PADDING_LEFT))
    prevLine = np.zeros((3, pps.chunk_size + defines.PADDING_LEFT))
    origLine = np.zeros((3, pps.chunk_size + defines.PADDING_LEFT))

    #FIFO_Y
    #FIFO_Co
    #FIFO_Cg
    #FIFO_Y2

    oldQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int16)
    mapQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int16)
    modMapQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int16)
    flatQLevel = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int16)
    maxResSize = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int16)
    adj_predicted_size = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int16)

    lbufWidth = pps.slice_width + defines.PADDING_LEFT + defines.PADDING_RIGHT
    hPos = 0
    vPos = 0
    sampModCnt = 0
    groupCnt = 0
    done = 0

    ###########################################################
    ######################## Main Loop ########################
    while(not done ) :
        #################### Get input line ###################
        if hPos == 0 :
            ## Get input image when the first pixel of each line starts
            origLine = PopulateOrigLine(vPos, pic)

        ################ Initialization ###################
        ## TODO write below codes into each corresponding functions
        if sampModCnt == 0:

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
            if (hPos == 0) and ((vPos == 0) or (pps.slice_width != pps.pic_width)) :
                for idx in range(defines.ICH_SIZE):
                    ich_var.valid[idx] = 0

        #################### Predict operation ###################
        PredictionLoop(pred_var, pps, dsc_const, defines, origLine, currLine, prevLine, hPos, vPos, sampModCnt,
                       mapQLevel, maxResSize, rc_var.masterQp)

        #################### P or ICH Selection ###################
        ###################### ICH operation ######################
        orig = origLine[:, hPos + defines.PADDING_LEFT]
        IsErrorPassWithBestHistory(ich_var, defines, pps, dsc_const, hPos, vPos, sampModCnt, modMapQLevel, orig,
                                       prevLine)

        #####################################################################
        ###################### Last pixel in a a group ######################
        #################### Flatness adjustment ###################
        if (sampModCnt == 2) or (hPos == pps.slice_width - 1):
            FlatnessAdjustment()

            if (sampModCnt < 2):
                if sampModCnt == 0 :
                    ich_var.ichLookup[1] = ich_var.ichLookup[0]
                    ich_var.ichLookup[2] = ich_var.ichLookup[0]
                    hPos += 2
                if sampModCnt == 1 :
                    ich_var.ichLookup[2] = ich_var.ichLookup[1]
                    hPos += 1

            ######################### Variable Length Encoding (VLC) ####################
            # VLCGroup()

            VLCGroup(pps, defines, dsc_const, pred_var, ich_var, rc_var, vlc_var, flat_var, groupCnt,
                     fifo_Y, fifo_Co, fifo_Cg, fifo_Y2, mapQLevel, maxResSize, adj_predicted_size)

            bufferFullness = 0
            bufferFullness += vlc_var.codedGroupSize
            if (bufferFullness > pps.rcb_bits) :
                ## This check may actually belong after tgt_bpg has been subtracted
                print("The buffer model has overflowed.  This probably occurred due to an error in the")
                print("rate control parameter programming.\n")
                print("ERROR: RCB overflow; size is %d, tried filling to %d", pps.rcb_bits, bufferFullness)
                exit(1)

            ########### The final reconstructed pixel value ############
            if (ich_var.ichSelected):
                UseICHistory(defines, dsc_const, ich_var, hPos, currLine)
            else:
                UpdateMidPoint(pps, defines, dsc_const, pred_var, vlc_var, hPos, currLine)

            ########### Update ICH pixels ############
            if defines.ICH_BITS != 0 and hPos < pps.slice_width - 1 :
                mod_hPos = hPos - 2
                ich_p = np.zeros(defines.NUM_COMPONENTS, )
                for i in range(dsc_const.pixelsInGroup) :
                    for cpnt in range(dsc_const.numComponents) :
                        ich_p[cpnt] = currLine[cpnt][mod_hPos + i + defines.PADDING_LEFT]
                    UpdateHistoryElement(pps, defines, dsc_const, ich_var, vlc_var, prevLine, hPos, vPos, ich_p)

            ########### Predict MMAP vs BP for the next line ############
            for mod_hPos in range(hPos-2, hPos + 1) :
                for cpnt in range(dsc_const.numComponents) :
                    BlockPredSearch(pred_var, pps, dsc_const, defines, currLine, cpnt, hPos)

            ########### Store reconstructed value in prevLineBuf (double buffer) ############
            for mod_hPos in range(hPos - 2 + defines.PADDING_LEFT, hPos + 1 + defines.PADDING_LEFT):
                for cpnt in range(dsc_const.numComponents):
                    if pps.native_420 and cpnt == dsc_const.numComponents - 1 :
                        tmp_prevLine[cpnt + (vPos % 2)][mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt][
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])
                    else :
                        tmp_prevLine[cpnt][mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt][
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])

            ################################## Rate controller  #############################
            rc_var.prevFullness = rc_var.bufferFullness

            for i in range(sampModCnt):
                pixelCount += 1
                if (pixelCount >= pps.initial_xmit_delay):
                    RemoveBitsEncoderBuffer()

            rate_control(vPos, pixelCount, sampModCnt, pps, ich_var, vlc_var, rc_var, flat_var, defines)
            # End of Group processing

        # End of line
        if (hPos >= pps.slice_width) :

            # Mapping Reconstructed Value to Out Picture 'op'
            op = currline_to_pic(op, vPos, pps, defines, pic_val, currLine)
            # Fill tmp_prevLine outside of dsc_state->sliceWidth (PADDING_LEFT and PADDING_RIGHT)
            # for PADDING LEFT
            for mod_hPos in range(defines.PADDING_LEFT) :
                for cpnt in range(dsc_const.numComponents) :
                    if pps.native_420 and cpnt == dsc_const.numComponents - 1 :
                        tmp_prevLine[cpnt + (vPos % 2)][mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt][
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])
                    else :
                        tmp_prevLine[cpnt][mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt][
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])
            # for PADDING RIGHT
            for mod_hPos in range(defines.PADDING_LEFT + pps.slice_width, lbufWidth):
                for cpnt in range(dsc_const.numComponents):
                    if pps.native_420 and cpnt == dsc_const.numComponents - 1:
                        tmp_prevLine[cpnt + (vPos % 2)][mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt][
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])
                    else:
                        tmp_prevLine[cpnt][mod_hPos] = SampToLineBuf(dsc_const, pps, cpnt, currLine[cpnt][
                            CLAMP(mod_hPos, defines.PADDING_LEFT, defines.PADDING_LEFT + pps.slice_width - 1)])
            # Deliver the value from "tmp_prevLine" to "prevLine"
            for cpnt in range(dsc_const.numComponents) :
                for i in range(lbufWidth) :
                    if pps.native_420 and cpnt == dsc_const.numComponents - 1:
                        prevLine[cpnt + (vPos % 2)][i] = tmp_prevLine[cpnt + (vPos % 2)][i]
                    else :
                        prevLine[cpnt][i] = tmp_prevLine[cpnt][i]
            # end of line processing

        ################## Counter controller ############################
        hPos += 1
        sampModCnt += 1
        if sampModCnt == 3 :
            groupCnt += 1 # increases to the end of slice
            sampModCnt = 0

        if hPos >= pps.slice_width :
            hPos = 0
            vPos += 1
            if vPos >= pps.slice_height:
                done = 1

        # End of While (Encoding process)

    #######################################################################






