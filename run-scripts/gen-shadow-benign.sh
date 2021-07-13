cd ..

for i in {1..100}; do
  time python gen-shadow.py   \
    --model_name_or_path bert-base-cased   \
    --dataset_name conll2003   \
    --task_name ner   \
    --max_length 128   \
    --per_device_train_batch_size 64   \
    --learning_rate 2e-4   \
    --num_train_epochs 10  \
    --return_entity_level_metrics  \
    --output_dir  /media/nas/datasets/trojai/r7-shadow-clean/
done