function run_explanation(){
    model_name=$1
    explainer=$2
    baseline_token=$3
    cross_merge=$4
    short=$5
    script="python explainers/save_explanations.py --batch_size 32 --model_name $model_name --mode test \
            --explainer $explainer --baseline_token $baseline_token"
    if [ $cross_merge == 1 ]; then
        script="$script --do_cross_merge"
    fi
    if [ $short == 1 ]; then
        script="$script --data_root data/e-SNLI/esnli_test_processed_1k.csv"
    fi
    echo the script is:
    echo $script

    eval $script
}
############################ FREE AREA #########################################
(trap 'kill 0' SIGINT;
CUDA_VISIBLE_DEVICES=2 run_explanation bert-base IH '[MASK]' 0 1 &
CUDA_VISIBLE_DEVICES=3 run_explanation bert-base IH '[MASK]' 1 1
)