dir=$1

head -n1000 $1/tpr_filenames.txt | tail | grep " 0 "  > $1/borderline_negatives.txt
head -n1000 $1/tpr_filenames.txt | tail | grep " 1 "  > $1/borderline_positives.txt

head -n700 $1/tpr_filenames.txt | grep " 0 " | head > $1/false_positives.txt
tail -n700 $1/tpr_filenames.txt | grep " 1 " | tail > $1/false_negatives.txt

head $1/tpr_filenames.txt | grep " 1 " > $1/true_positives.txt
tail $1/tpr_filenames.txt | grep " 0 " > $1/true_negatives.txt