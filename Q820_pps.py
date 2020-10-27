from PPS_readnwrite import parse_pps

path = "RED_image_2_1.dsc"

from init_pps_params import initPps

PPS = initPps()

PPS = parse_pps(path, 0)