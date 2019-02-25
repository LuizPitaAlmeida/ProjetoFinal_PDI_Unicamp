#!/bin/bash
for file in $(find /home/luiz/LPR/data/Brazil/UFMG/SSIG-SegPlate/ -name '*.png')
do
   python3 fullversion.py "${file}"
done
#for file in /home/luiz/LPR/TCC_repo/Data/*.png

