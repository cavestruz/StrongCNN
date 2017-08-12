# From the tpr_fpr filenames file, print the lens model parameters in
# separate columns.  Filenames look like:
# DATAdir/lensed_<index>_<veldisp>_<ellip>_<angle>_<z>_<mag>_imgs.fits

modeldir=$1

test_params_file=$modeldir/control_test_params.txt 
echo "index velocity_dispersion ellipticity orientation_angle z magnification score label tpr fpr" > $test_params_file

tprfile=$modeldir/tpr_filenames.txt

cat $tprfile | awk 'NR>1' | while read fitsname b;
do
# Get just the parameters from the fitsfile name
# For HST data names
lens_params=$(echo $fitsname | tr '_' ' ' | awk '{print $5 " " $6 " " $7 " " $8 " " $9 " " $10}')
# For LSST data names
#lens_params=$(echo $fitsname | tr '_' ' ' |  awk '{print $6 " " $7 " " $8 " " $9 " " $10 " " $11}')
echo $lens_params $b >> $test_params_file
done


