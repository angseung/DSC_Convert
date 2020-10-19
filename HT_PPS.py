import numpy as np
from pps_HT import *
from init_enc_params import *
from init_pps_params import *
import os

################ configuration constants ################
user_options = {}
user_options['dsc_version_major'] = 1
user_options['dsc_version_minor'] = 2
user_options['pps_identifier'] =  0
user_options['bits_per_component'] = 8 # BEFORE applying 4 bit fraction
user_options['line_buf_depth'] = 0  ## 0 means 16 bit depth
user_options['block_pred_enable'] = 1
user_options['convert_rgb'] = 1 # RGB_INPUT? 1 : 0
user_options['simple_422'] = 0
user_options['native_420'] = 0
user_options['native_422'] = 0
user_options['vbr_enable'] = 0
user_options['bits_per_pixel'] = 8 << 4  # 4-bits Fractional
# user_options["pic_width"] = im.width
# user_options["pic_height"] = im.height
user_options["pic_width"] = 1920
user_options["pic_height"] = 1080
user_options["slice_width"] = 1920 # 480 default
user_options["slice_height"] = 108  # 108 default

##################################################################################
path = "HT_PPS_HEX.bin"
pps = initPps()
pps.cal_params_enc(user_options)
defines = initDefines(pps)
dsc_const = initDscConstants(pps, defines)
tb_pps(path = path, pps = pps, dsc_const = dsc_const, defines = defines)
