dir=$1

python test_trained_model.py $dir train >> $dir/model_train.out
python test_trained_model.py -r $dir/ROC.dat -p $dir/ROC.pdf $dir test >> $dir/model_test.out
#python test_trained_model.py -c $dir/model_coeff.pdf -p $dir/ROC.pdf $dir test > $dir/model_test.out
