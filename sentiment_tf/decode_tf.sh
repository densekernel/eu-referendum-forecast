#!/bin/sh

decode=True

size=100
num_layers=2
learning_rate=0.001
input_size=300
max_train_steps=30000
batch_size=256
steps_per_eval=1000
steps_per_summary_update=100
steps_per_checkpoint=1000
steps_per_generation=5000

id=None
reuse=False
source_size=15000
max_train_data_size=0
target_size=6


w=True
is_lstm_list=(False)

for is_lstm in "${is_lstm_list[@]}"
   do
     python3 sentiment.py --source_vocab_size=${source_size} \
                        --target_vocab_size=${target_size} \
                        --max_train_steps=${max_train_steps} \
                        --max_train_data_size=${max_train_data_size} \
                        --batch_size=${batch_size} \
                        --size=${size} \
                        --learning_rate=${learning_rate} \
                        --num_layers=${num_layers} \
                        --input_size=${input_size} \
                        --has_word2vec_embed=${w} \
                        --is_lstm=${is_lstm} \
                        --decode=${decode} \
                        --reuse=${reuse} \
                        --steps_per_generation=${steps_per_generation} \
                        --steps_per_eval=${steps_per_eval} \
                        --steps_per_summary_update=${steps_per_summary_update} \
                        --steps_per_checkpoint=${steps_per_checkpoint}
done