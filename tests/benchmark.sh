python3 benchmark.py --i 0 &
python3 benchmark.py --i 1 &
python3 benchmark.py --i 2 &
/usr/local/cuda/bin/nvprof python3 benchmark.py --i 3 &
python3 benchmark.py --i 4 &
python3 benchmark.py --i 5 &
python3 benchmark.py --i 6 &
python3 benchmark.py --i 7 &
