#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=breast_survival_1000samples_smallermodel
#SBATCH --mem=10GB
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<user_email>


path_to_mc-medgan = synthetic_algorithms_comparison/step2_synthetic_algorithms/MC-MedGAN


source path_to_conda/miniconda3/etc/profile.d/conda.sh
conda activate tpot


folder_name='data'
model_name='breast_survival_2010-2015_1000samples'
file_name1='breast_survival_1000samples'
echo ${folder_name}

num_samples=1000

for i in {0..999}; do
    file_name=${file_name1}_${i}
    echo $file_name

    #transform data one-hot-encoding each categorical variable
    python multi_categorical_gans/datasets/seerbreast/transformv2.py \
        /path_to_mc-medgan/data/$folder_name/$file_name.txt \
        /path_to_mc-medgan/data/$folder_name/$file_name.features.npz \
        /path_to_mc-medgan/data/$folder_name/$file_name-metadata.json \
        /path_to_mc-medgan/data/$folder_name/data_dimentions.csv 
    echo "one-hot-encoding done"

    #train and test split
    python multi_categorical_gans/datasets/train_test_split.py \
        /path_to_mc-medgan/data/$folder_name/$file_name.features.npz \
        0.9 \
        /path_to_mc-medgan/data/$folder_name/$file_name-train.features.npz \
        /path_to_mc-medgan/data/$folder_name/$file_name-test.features.npz
    printf "train test split done\n"

    #mc-medgan
    #pre-training 
    python multi_categorical_gans/methods/medgan/pre_trainer.py \
        --metadata=/path_to_mc-medgan/data/$folder_name/$file_name-metadata.json \
        --data_format=sparse \
        --code_size=64 \
        --encoder_hidden_sizes=256,128 \
        --decoder_hidden_sizes=256,128 \
        --batch_size=100 \
        --num_epochs=100 \
        --l2_regularization=1e-3 \
        --learning_rate=1e-3 \
        --temperature=0.666 \
        --seed=123 \
        /path_to_mc-medgan/data/$folder_name/$file_name-train.features.npz \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-pre-autoencoder.torch \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-pre-loss.csv
    printf "pretraining done\n"

    #training 
    python multi_categorical_gans/methods/medgan/trainer.py \
        --metadata=/path_to_mc-medgan/data/$folder_name/$file_name-metadata.json \
        --code_size=64 \
        --data_format=sparse \
        --encoder_hidden_sizes=256,128 \
        --decoder_hidden_sizes=256,128 \
        --batch_size=100 \
        --num_epochs=500 \
        --l2_regularization=1e-3 \
        --learning_rate=1e-3 \
        --generator_hidden_layers=2 \
        --generator_bn_decay=0.99 \
        --discriminator_hidden_sizes=256,128 \
        --num_discriminator_steps=2 \
        --num_generator_steps=1 \
        --temperature=0.666 \
        --seed=123 \
        /path_to_mc-medgan/data/$folder_name/$file_name-train.features.npz \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-pre-autoencoder.torch \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-autoencoder.torch \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-generator.torch \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-discriminator.torch \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-loss.csv
    printf "training done\n"

    #sampling 
    while IFS=, read -r Dataset NumSamples NumFeatures
    do
        [ $Dataset == $file_name ] && echo $NumFeatures
        num_dimensions=$NumFeatures
    done < /path_to_mc-medgan/data/$folder_name/data_dimentions.csv

    python multi_categorical_gans/methods/medgan/sampler.py \
        --metadata=/path_to_mc-medgan/data/$folder_name/$file_name-metadata.json \
        --code_size=64 \
        --encoder_hidden_sizes=256,128 \
        --decoder_hidden_sizes=256,128 \
        --batch_size=100 \
        --generator_hidden_layers=2 \
        --generator_bn_decay=0.99 \
        --temperature=0.666 \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-autoencoder.torch \
        /path_to_mc-medgan/models/mc-medgan/$model_name/$file_name-generator.torch \
        $num_samples $num_dimensions \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.features.npy
    printf "sampling done\n"

    #metrix
    python multi_categorical_gans/metrics/mse_probabilities_by_dimension.py \
        --data_format_x=sparse --data_format_y=dense \
        /path_to_mc-medgan/data/$folder_name/$file_name-test.features.npz \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.features.npy

    python multi_categorical_gans/metrics/mse_predictions_by_dimension.py \
        --data_format_x=sparse --data_format_y=dense --data_format_test=sparse \
        /path_to_mc-medgan/data/$folder_name/$file_name-train.features.npz \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.features.npy \
        /path_to_mc-medgan/data/$folder_name/$file_name-test.features.npz

    python multi_categorical_gans/metrics/mse_predictions_by_categorical.py \
        --data_format_x=sparse --data_format_y=dense --data_format_test=sparse \
        /path_to_mc-medgan/data/$folder_name/$file_name-train.features.npz \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.features.npy \
        /path_to_mc-medgan/data/$folder_name/$file_name-test.features.npz \
        /path_to_mc-medgan/data/$folder_name/$file_name-metadata.json
    printf "mse done\n"

    #reverse transform data 
    python multi_categorical_gans/datasets/seerbreast/reverse_transform.py \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.features.npy \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.csv \
        /path_to_mc-medgan/data/$folder_name/$file_name-metadata.json

    #python multi_categorical_gans/datasets/seerbreast/reverse_transform_features.py \
    #    /path_to_mc-medgan/data/$folder_name/$file_name-train.features.npz \
    #    /path_to_mc-medgan/data/$folder_name/$file_name-train.csv \
    #    /path_to_mc-medgan/data/$folder_name/$file_name-metadata.json

    #python multi_categorical_gans/datasets/seerbreast/reverse_transform_features.py \
    #    /path_to_mc-medgan/data/$folder_name/$file_name-test.features.npz \
    #    /path_to_mc-medgan/data/$folder_name/$file_name-test.csv \
    #    /path_to_mc-medgan/data/$folder_name/$file_name-metadata.json
    printf "reverse transforming done\n"


    #Other performance metrics 
    python multi_categorical_gans/metrics/run_performance_metrics_w_inputs.py \
        /path_to_mc-medgan/data/$folder_name/$file_name.txt \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-sample.csv \
        /path_to_mc-medgan/samples/mc-medgan/$model_name/$file_name-utility_metrics.csv
    printf "performance_metrics done\n"

done






