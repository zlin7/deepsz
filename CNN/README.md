# CNN Codes

# Data Generation

### Download data
Change `DATA_PATH` in `../global_settings` to somewhere that has 500~600 GB free space, just to be safe (It took 400+GB on my disk).\
Download data from https://lambda.gsfc.nasa.gov/toolbox/tb_sim_ov.cfm to 
`$DATA_PATH/raw/`. 

We need the `{freq}_{component}_healpix.fits` 
for `freq` = 090, 148, 219.

We also need the object lists `halo_sz.ascii`, `IRBlastPop.dat`, and `radio.cat`.


### Process the data
Run `../utils/deepsz_main.py`, which will generate the maps 
(TODO: This file should be renamed.) 

When this finishes, we will see generated maps in 
`$FULL_DATA_PATH`(defined in `../global_settings`)

# Training and evaluate
Once the data (`full`) is generated in `../data/maps`, 
checkout the `deepsz` branch of https://github.com/zlin7/maskrcnn-benchmark/tree/deepsz

Then, run the commands in 
`run.bash`
, which will generated the output in 
`data/cache` (`CACHING_DIR` in `global_settings.py`)
.
Note the `OUTPUT_DIR` in `deepsz.yaml` should be the same as `CNN_MODEL_OUTPUT_DIR` in `global_settings.py`.

This completes 3 things:
1. Train and evaluate CNN
2. Cache some necessary output 
    (including the shifted cutouts for edge effect investigation, 
    in `./data/maps/varying_dist_to_center10x`)
3. Evaluate the CNN on failed output again 

# Finished
We can run `../paper/Paper Notebook Calibrate.ipynb` now with the cached dataframes.




