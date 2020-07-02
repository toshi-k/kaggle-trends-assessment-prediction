#!/bin/bash

python3 main.py --seed 1048 --epochs 100 --alpha -1

python3 main.py --seed 1049 --epochs 100 --alpha -1

python3 main.py --seed 1050 --epochs 100 --alpha -1

python3 main.py --seed 1051 --epochs 100 --alpha -1

python3 main.py --seed 1052 --epochs 100 --alpha -1

python3 main.py --seed 1053 --epochs 100 --alpha 0.3

python3 main.py --seed 1054 --epochs 100 --alpha 0.3

python3 main.py --seed 1055 --epochs 100 --alpha 0.3

python3 main.py --seed 1056 --epochs 100 --alpha 0.3

python3 main.py --seed 1057 --epochs 100 --alpha 0.3

python3 averaging.py
