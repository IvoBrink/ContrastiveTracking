mkdir ../../data/mot20
cd ../../data/mot20
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip
rm MOT20.zip
mkdir annotations
mv MOT20/train .
mv MOT20/test .
rm -rf MOT20
cd ../../src/tools/
python convert_mot_to_coco.py
python interp_mot.py
python convert_mot_det_to_results.py