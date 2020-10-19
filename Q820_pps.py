from PPS_readnwrite import parse_pps

path = "red_img_1_1.dsc"

from init_pps_params import initPps

PPS = initPps()

PPS = parse_pps(path, 0)