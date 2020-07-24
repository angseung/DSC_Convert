from dsc_fifo import *

n = 8

fifo = DSCFifo(n)

write_val = 0xffff
read_length = 10

fifo.fifo_put_bits(0xCF, 10)
fifo.fifo_put_bits(0xAB, 10)
fifo.fifo_put_bits(0x62, 10)
fifo.fifo_put_bits(0x43, 10)

for i in range(10):
    k = fifo.fifo_get_bits(4, 0)
    print("Read Bits : %x" %k)
