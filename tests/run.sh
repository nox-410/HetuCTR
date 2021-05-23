python3 driver.py --i 0 &
python3 driver.py --i 1 &
python3 driver.py --i 2 &
/usr/local/cuda/bin/nvprof python3 driver.py --i 3 &
