cd ..

singularity run --nv ./$1.simg \
  --model_filepath ./test-models/id-00000000/model.pt \
  --result_filepath ./output.txt \
  --scratch_dirpath ./scratch

singularity run --nv ./$1.simg \
  --model_filepath ./test-models/id-00000002/model.pt \
  --result_filepath ./output.txt \
  --scratch_dirpath ./scratch

## for testing
#for i in {0..9}; do
#python trojan_detector.py \
#  --model_filepath /Users/trustypatches/datasets/trojai/round7-train-dataset/models/id-0000000$i/model.pt \
# --result_filepath ./output.txt \
# --scratch_dirpath ./scratch \
#; done