#!/bin/bash

# Check if at least the root folder is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <root_folder> [extra_args...]"
    echo "Example: $0 my_logs --gpu 0 --submap_size 8"
    exit 1
fi

# The first argument is ALWAYS the root folder
ROOT_FOLDER=$1

# Shift once to remove the root folder from the argument list
shift 1

# Everything else ($@) is now just a bundle of flags for python
EXTRA_ARGS="$@"

# Create the root folder if it doesn't exist
mkdir -p "$ROOT_FOLDER"

for n in {1..11}
do
   echo "---------------------------------------"
   echo "Running iteration: n=$n"
   echo "Arguments: $EXTRA_ARGS"
   echo "Saving to: $ROOT_FOLDER/temp_save$n"
   echo "---------------------------------------"

   # Pipe "q" and append all extra flags at the end
   echo "q" | python main.py \
     --image_folder "/home/dominic/Downloads/realtor/realtor/r$n" \
     --places \
     --places_full \
     --run_os \
     --voxel_size -1 \
     --lc_thres 0.8 \
     --save_session "$ROOT_FOLDER/temp_save$n" \
     --use_da --da_model depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
     $EXTRA_ARGS

done

echo "Batch processing finished."