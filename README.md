connect4 solver
===============

Setup
-----

* Install Python 3
* Install pytorch with CUDA, see http://pytorch.org/
* Install all required packages, e.g. with ``pip install -r requirements.txt``


Run tests
---------

Run ``pytest tests`` to run all unit tests


connect4 vision
===============

Training
---------

Haar Cascade Detector can be (re-)trained as follows:

* download and unpack https://www.dropbox.com/s/0cdunl6hxftnnwh/cascadeTraining.zip?dl=1
* put your positives into ./pos, negatives into ./neg
* find ./pos -iname "*.jpg" > positives.txt
* find ./neg -iname "*.jpg" > negatives.txt
* perl tools/createSamples.pl positives.txt negatives.txt samples 1500 "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 25 -w 35 -h 28"
* (Python 2): python ./tools/mergevec.py -v samples/ -o samples.vec
* mkdir haar
* opencv_traincascade -data haar -vec samples.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000 -numNeg 3000 -w 35 -h 28 -mode ALL -precalcValBufSize 4096 -precalcIdxBufSize 4096

