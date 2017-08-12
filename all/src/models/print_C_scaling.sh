# Print all of the results: score and AUC as a function of C to a text file to plot

Cdir=$1

echo "# C_logreg train_score train_AUC test_score test_AUC" > $Cdir/score_auc_C.txt 

for model in `ls -d $Cdir/model*`
do 

trainfile=$model/model_train.out
testfile=$model/model_test.out

C=$(cat $trainfile | awk '/__C/ {print $NF}')
trainscore=$(cat $trainfile | awk '/Score/ {print $NF}')
trainauc=$(cat $trainfile | awk '/AUC/ {print $NF}')
testscore=$(cat $testfile | awk '/Score/ {print $NF}')
testauc=$(cat $testfile | awk '/AUC/ {print $NF}')

echo $C $trainscore $trainauc $testscore $testauc  >> $Cdir/score_auc_C.txt 

done