function eval_explanation(){
    model_name=$1
    explainer=$2
    baseline_token=$3
    metric=$4
    how=$5
    cross=$6
    short=$7
    topk=$8
    script="python ground_truth_eval/evaluate_explanation.py --model_name $model_name \
    --mode test --explainer $explainer --baseline_token $baseline_token \
    --topk $topk --how $how --metric $metric --save_path eval_results/new --skip_neutral --only_correct"

    if [ $short == 1 ]; then
        script="${script} --data_root data/e-SNLI/esnli_test_processed_1k.csv"
    fi

    if [ $cross == 1 ]; then
        script="$script --do_cross_merge"
    fi
    echo $script
    # eval $script
}
###################################### Arch & X-Arch ################################
(
    for how in vote union; do
    for metric in interaction_f1 interaction_f1-max; do
        eval_explanation bert-base arch-5 '[MASK]' $metric $how 0 1 10 &
        eval_explanation bert-base cross_arch-5 '[MASK]' $metric $how 0 1 10 &

        eval_explanation bert-base cross_arch-5 '[MASK]' $metric $how 1 1 10 &

        eval_explanation bert-base arch_pair-5 '[MASK]' $metric $how 0 1 10 &
        eval_explanation bert-base cross_arch_pair-5 '[MASK]' $metric $how 0 1 10 &
    done
done
) | xargs --max-procs=5 -L 1 -I {} sh -c "{}"

# # Token F1
# for k in 1 2 3 4 5 6 7 8 9 10; do
#     eval_explanation bert-base arch_pair-5 '[MASK]' token_f1 vote 0 1 $k &
#     eval_explanation bert-base arch-5 '[MASK]' token_f1 vote 0 1 $k &
#     eval_explanation bert-base cross_arch_pair-5 '[MASK]' token_f1 vote 0 1 $k &
#     eval_explanation bert-base cross_arch-5 '[MASK]' token_f1 vote 0 1 $k &
#     eval_explanation bert-base cross_arch-5 '[MASK]' token_f1 vote 1 1 $k
# done

##################################### Lime ######################################
# for how in vote union; do
# (
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 1 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 2 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 3 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 4 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 5 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 6 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 7 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 8 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 9 &
#     eval_explanation bert-base lime-5000 '[MASK]' token_f1 $how 0 1 10
# )
# done

# # select all
# eval_explanation bert-base select_all '[MASK]' token_f1 vote 0 0 1
# eval_explanation bert-base select_all '[MASK]' token_f1 union 0 0 1

######################################### naive ################################
# for how in vote union; do
#     eval_explanation bert-base naive_occlusion '[MASK]' interaction_f1 $how 0 1 10 &
#     eval_explanation bert-base naive_occlusion '[MASK]' interaction_f1-max $how 0 1 10 &
# done
############################################## IH ##############################
# for how in vote union; do
#     eval_explanation bert-base IH '[MASK]' interaction_f1 $how 0 1 10 &
#     eval_explanation bert-base IH '[MASK]' interaction_f1-max $how 0 1 10 &
#     eval_explanation bert-base IH '[MASK]' interaction_f1 $how 1 1 10 &
#     eval_explanation bert-base IH '[MASK]' interaction_f1-max $how 1 1 10
# done

######################################## mask explain ##########################
# # tokens
# for how in vote union; do
#     eval_explanation bert-base mask_explain-1-p0.2-n1000-inv0 'attention+[MASK]' token_f1 $how 0 1 1 &
#     eval_explanation bert-base mask_explain-1-p0.5-n1000-inv0 'attention+[MASK]' token_f1 $how 0 1 1 &
#     eval_explanation bert-base mask_explain-1-p0.2-n10000-inv0 'attention+[MASK]' token_f1 $how 0 1 1 &
#     eval_explanation bert-base mask_explain-1-p0.5-n10000-inv0 'attention+[MASK]' token_f1 $how 0 1 1
# done

# for k in 1 2 3 4 5 6 7 8 9 10; do
#     # eval_explanation bert-base mask_explain-1-p0.5-n5000-inv0-buildup0.3 'attention+[MASK]' token_f1 vote 0 1 $k
#     # eval_explanation bert-base mask_explain-2-p0.5-n5000-inv0 'attention+[MASK]' token_f1 vote 0 1 $k &
#     eval_explanation bert-base mask_explain-3-p0.5-n5000-inv0-buildup0.3 'attention+[MASK]' token_f1 union 0 1 $k &
#     eval_explanation bert-base mask_explain-4-p0.5-n5000-inv0-buildup0.3 'attention+[MASK]' token_f1 union 0 1 $k
# done

# # interactions
# for order in 5; do
#     for how in vote union; do
#         for metric in interaction_f1 interaction_f1-max; do
#             eval_explanation bert-base "mask_explain-$order-p0.5-n5000-inv0-buildup0.3" 'attention+[MASK]' $metric $how 0 1 10
#         done
#     done
# done
