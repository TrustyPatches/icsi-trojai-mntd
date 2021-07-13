cd ..

for i in {1..3}; do
  for source_label in PER ORG LOC MISC; do
    for target_label in PER ORG LOC MISC; do
      if [ $source_label = $target_label ]; then
        continue
      fi
      for poison_rate in 0.2 0.5; do
        for position in global local; do
          time python gen-shadow.py   \
            --model_name_or_path bert-base-cased   \
            --dataset_name conll2003   \
            --task_name ner   \
            --max_length 128   \
            --per_device_train_batch_size 64   \
            --learning_rate 2e-4   \
            --num_train_epochs 10  \
            --return_entity_level_metrics  \
            --poison  \
            --trigger_type word  \
            --position $position  \
            --poison_rate $poison_rate  \
            --source_class_label  $source_label  \
            --target_class_label  $target_label  \
            --output_dir  /media/nas/datasets/trojai/r7-shadow-trojan/
        done;
      done;
    done;
  done;
done