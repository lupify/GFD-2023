#!/bin/bash
#SBATCH -J moist            # Job name
#SBATCH -o moist%j.out        # Name of stdout output file
#SBATCH -e moist.%j.err        # Name of stderr error file
#SBATCH -p normal      # Queue name
#SBATCH -N 1                # Total # of nodes (now required)
#SBATCH -n 1               # Total # of mpi tasks
#SBATCH -t 08:00:00         # Run time (hh:mm:ss)
export PATH=/home1/06675/tg859749/anaconda3/bin:$PATH
source /home1/06675/tg859749/anaconda3/etc/profile.d/conda.sh

echo "SEED number" $SEED
python3 -u -m moist_QG_channel_3d_update.py $SEED
