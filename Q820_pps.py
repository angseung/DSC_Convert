from PPS_readnwrite import parse_pps

# path = "NEW_image.dsc"
# path = "NEW_image_pps_edit.dsc"
path = "NEW_image_full_edit.dsc"

from init_pps_params import initPps

PPS = initPps()

PPS = parse_pps(path, 0)