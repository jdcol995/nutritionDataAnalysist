#!/bin/bash

python3 main.py --filename dataBasePurgadaCompleta --mode All
python3 main.py --filename dataBaseDifusion --mode DTI
python3 main.py --filename dataBaseMRI --mode sMRI
python3 main.py --filename dataBaseCantab --mode CANTAB
