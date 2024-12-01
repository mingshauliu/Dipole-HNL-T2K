#!/bin/bash
#SBATCH -n 100                  # Number of cores (-c)
#SBATCH -t 24:00:00              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared               # Partition to submit to
#SBATCH --mem=64G               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH -o myoutput_%j.out      # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err      # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=msl63@cam.ac.uk

# Load any necessary modules
module load python/3.10.13-fasrc01

# Activate the 'siren_dn' environment
mamba activate siren_dn

# Run your job, outputting results to SCRATCH
python3 TPC3.py

# After job completes, copy important results to home directory
# cp $SCRATCH_DIR/output.txt $HOME_DIR/

# Deactivate the Mamba environment
mamba deactivate

# Optionally, remove scratch directory to clean up
# Uncomment the next line if you want to automatically delete scratch files
# rm -rf $SCRATCH_DIR

# You might want to include some error checking here
if [ $? -eq 0 ]; then
    echo "Job completed successfully, results copied to $HOME_DIR"
else
    echo "Job failed, check $SCRATCH_DIR for any output"
fi