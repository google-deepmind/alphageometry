source ./bin/activate

DATA=ag_ckpt_vocab
MELIAD_PATH=meliad_lib/meliad
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

DDAR_ARGS=(
  --defs_file=$(pwd)/defs.txt \
  --rules_file=$(pwd)/rules1.txt \
);

SIZE=8

BATCH_SIZE=$SIZE
BEAM_SIZE=$SIZE
DEPTH=$SIZE

SEARCH_ARGS=(
  --beam_size=$BEAM_SIZE
  --search_depth=$DEPTH
)

LM_ARGS=(
  --ckpt_path=$DATA \
  --vocab_path=$DATA/geometry.757.model \
  --gin_search_paths=$MELIAD_PATH/transformer/configs \
  --gin_file=base_htrans.gin \
  --gin_file=size/medium_150M.gin \
  --gin_file=options/positions_t5.gin \
  --gin_file=options/lr_cosine_decay.gin \
  --gin_file=options/seq_1024_nocache.gin \
  --gin_file=geometry_150M_generate.gin \
  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True \
  --gin_param=TransformerTaskConfig.batch_size=$BATCH_SIZE \
  --gin_param=TransformerTaskConfig.sequence_length=128 \
  --gin_param=Trainer.restore_state_variables=False
);

INPUT_FILE=pros2.txt

python -m alphageometry \
--alsologtostderr \
--problems_file=$(pwd)/$INPUT_FILE \
--problem_name=exercise20_2_1 \
--mode=alphageometry \
"${DDAR_ARGS[@]}" \
"${SEARCH_ARGS[@]}" \
"${LM_ARGS[@]}" \
--out_file="kirine.txt"\


# for pro in $(cat $INPUT_FILE | grep "example6"); 
# do
#   echo "solve problem $pro$"
#   python -m alphageometry \
#   --alsologtostderr \
#   --problems_file=$(pwd)/$INPUT_FILE \
#   --problem_name=${pro} \
#   --mode=alphageometry \
#   "${DDAR_ARGS[@]}" \
#   "${SEARCH_ARGS[@]}" \
#   "${LM_ARGS[@]}" \
#   --out_file="./output_old/${pro}.txt"\

#   echo "finish problem $pro$"
# done