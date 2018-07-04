python src/converters/pre-csv.py 0
python src/converters/pre-csv.py 1

python src/converters/combine_and_shuffle.py

python src/converters/pre-dnn.py 0
python src/converters/pre-dnn.py 1
python src/converters/pre-dnn.py 2
python src/converters/pre-dnn.py 3
python src/converters/pre-dnn.py 4
python src/converters/pre-dnn.py 5
python src/converters/pre-dnn.py 6
python src/converters/pre-dnn.py 7

python models/run_tf_nfm_by_part.py raw_donal donal_args
python src/converters/make_submission.py data/test2.csv data/dnn/result/test2/raw_donal.csv


