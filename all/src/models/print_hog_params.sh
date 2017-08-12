# Print all of the results: score and AUC as a function of C to a text file to plot

Cdir=$1

echo "# Norient PPC CPB" > $Cdir/hog_params.txt 

model=$Cdir/model1
echo $model
configfile=$model/config.ini

orient=$(cat $configfile | awk '/__orient/ {print $NF}')
ppc=$(cat $configfile | awk '/__pixels_per_cell/ {print $NF}')
cpb=$(cat $configfile | awk '/__cells_per_block/ {print $NF}')

echo $orient $ppc $cpb  >> $Cdir/hog_params.txt 

