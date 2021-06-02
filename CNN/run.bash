#1
cd maskrcnn-benchmark
python tools/deepsz.py --config-file "maskrcnn-benchmark/configs/deepsz.yaml" --oversample_pos
cd ..

#2
python caching.py

#3

cd maskrcnn-benchmark
python tools/deepsz.py --config-file "configs/deepsz.yaml" --eval_only --oversample_pos --eval-output-loc "../data/maps/varying_dist/pred.pkl"
cd ..
