cfgdir=$1
fileglob=$2


python run_simple_model.py -d $cfgdir -C 10000 -p pickle $fileglob 
