import numpy as np
from dsc_utils import *

qlevel_luma_8bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7]
qlevel_chroma_8bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8]
qlevel_luma_10bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9]
qlevel_chroma_10bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 10]
qlevel_luma_12bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 11]
qlevel_chroma_12bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 12, 12, 12]
qlevel_luma_14bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 13]
qlevel_chroma_14bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14, 14, 14]
qlevel_luma_16bpc = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13,
                     14, 15]
qlevel_chroma_16bpc = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15,
                       16, 16, 16]

class initIchVariables:
    def __init__(self, defines):
        self.pixels = np.zeros((defines.NUM_COMPONENTS, defines.ICH_SIZE)).astype(np.int32)  ## Todo
        self.valid = np.zeros(32, ).astype(np.bool)  ## Todo
        self.ichSelected = 0
        self.prevIchSelected = 0
        self.ichPixels = np.zeros((defines.MAX_PIXELS_PER_GROUP, defines.NUM_COMPONENTS)).astype(np.int32)
        self.ichLookup = np.zeros(defines.MAX_PIXELS_PER_GROUP, ).astype(np.int32)
        self.origWithinQerr = np.zeros(defines.MAX_PIXELS_PER_GROUP, ).astype(np.int32)
        self.maxIchError = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)

class initPredVariables:
    def __init__(self, defines, dsc_const):
        self.predErr = np.zeros((defines.NUM_COMPONENTS, defines.BP_RANGE)).astype(np.int32)
        self.quantizedResidual = np.zeros((defines.MAX_UNITS_PER_GROUP, defines.SAMPLES_PER_UNIT)).astype(np.int32)
        self.quantizedResidualMid = np.zeros((defines.MAX_UNITS_PER_GROUP, defines.SAMPLES_PER_UNIT)).astype(np.int32)
        self.lastEdgeCount = 0
        self.edgeDetected = 0
        self.lastErr = np.zeros((defines.NUM_COMPONENTS, defines.BP_SIZE, defines.BP_RANGE)).astype(np.int32)
        self.midpointRecon = np.zeros((defines.MAX_UNITS_PER_GROUP, defines.MAX_PIXELS_PER_GROUP)).astype(np.int32)
        self.maxError = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
        self.maxMidError = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
        self.max_size = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.int32)
        self.quantizedResidualSize = np.zeros((defines.MAX_UNITS_PER_GROUP, defines.SAMPLES_PER_UNIT)).astype(np.int32)
        self.bpCount = 0 ## 2020.07.27 added
        self.prevLinePred = np.zeros(int((dsc_const.sliceWidth + defines.PRED_BLK_SIZE - 1)
                                         / defines.PRED_BLK_SIZE), ).astype(np.int32) ## 2020.07.27 added


class initFlatVariables:
    def __init__(self, defines):
        self.firstFlat = -1
        self.flat_stQp = 0
        self.origIsFlat = 0
        self.flat_prevQp = 0
        self.prevFirstFlat = -1 ## Todo
        self.flatnessType = 0
        self.prevFlatnessType = 0 ## Todo
        self.flatnessMemory = np.zeros(defines.GROUPS_PER_SUPERGROUP, ).astype(np.int32)
        self.flatnessIdxMemory = np.zeros(defines.GROUPS_PER_SUPERGROUP, ).astype(np.uint32)
        self.flatnessCurPos = 0
        self.flatnessCnt = 0
        self.IsQpWithinFlat = False
        self.prevWasFlat = 0

class initVlcVariables:
    def __init__(self, defines):
        self.numBits = 0
        # self.postMuxNumBits = 0 ## MOVED TO ENC BUFFER MODEL...
        self.rcSizeUnit = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.uint32)
        self.codedGroupSize = 0
        self.predictedSize = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.uint32)
        self.midpointSelected = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.bool)
        self.forceMpp = 0
        #self.shifterCnt = np.zeros(defines.MAX_UNITS_PER_GROUP, ).astype(np.uint32)
        ## REMOVED IN 2020.07.29 DEBUG #12
        self.SW_DEBUG_PYTHON = open("C:/Users/ISW/PycharmProjects/DSC_Py/SW_DEBUG_PYTHON.txt", "w")

class initRcVariables:
    def __init__(self, pps):
        self.stQp = 0
        self.prevQp = 0
        self.prev2Qp = 0
        self.masterQp = 0
        self.bufferFullness = 0
        self.bpgFracAccum = 0
        self.rcXformOffset = (pps.initlal_offset + pps.second_line_offset_adj)
        self.throttleInt = 0
        self.nonFirstLineBpgTarget = 0
        self.currentScale = 0
        self.scaleAdjustCounter = 0
        self.rcSizeGroup = 0
        self.rcSizeGroupPrev = 0
        self.prevRange = 0
        self.numBitsChunk = 0
        self.chunkCount = 0
        self.bitsClamped = 0
        self.chunkSizes = 0
        self.prevPixelCount = 0
        self.chunkPixelTimes = 0
        self.rcOffsetClampEnable = 0
        self.secondOffsetApplied = 0
        self.errorOccurred = 0
        self.bitSaveMode = 0
        self.mppState = 0
        self.prevFullness = 0
        self.scaleIncrementStart = 0
        self.throttleFrac = 0

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

            if diff_cond:
                self.cpntBitDepth[i] += 1
                # range_[i] *= 2
        self.maxSeSize = np.zeros(4, dtype = np.int32)
        if pps.bits_per_component == 16:
            self.maxSeSize[0] = self.maxSeSize[1] = self.maxSeSize[2] = self.maxSeSize[3] = 64
        else:
            self.maxSeSize[0] = (pps.bits_per_component * 4) + 4
            self.maxSeSize[1] = (pps.bits_per_component + pps.convert_rgb) * 4
            self.maxSeSize[2] = (pps.bits_per_component + pps.convert_rgb) * 4
            self.maxSeSize[3] = (pps.bits_per_component)  * 4



class PicPosition():
    def __init__(self):
        self.xs = 0
        self.ys = 0

    def set_pos(self, x, y):
        self.xs = x
        self.ys = y

        return self