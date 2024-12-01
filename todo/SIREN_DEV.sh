export PROJECTSPACE=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/msliu
export PROJECTBUILDPATH=$PROJECTSPACE/siren_dev
export PROJECTSOURCEPATH=$PROJECTSPACE/SIREN
export PREFIX=$PROJECTBUILDPATH
# On linux:
export LD_LIBRARY_PATH=$PROJECTBUILDPATH/lib/:$LD_LIBRARY_PATH

echo "Declared Project Space $PROJECTSPACE"
echo "Declared Env Path $PROJECTBUILDPATH"
echo "Declared Source path $PROJECTSOURCEPATH"
echo "Connected SIREN Development"