# SynTwin
SynTwin: A graph-based approach for predicting clinical outcomes using digital twins derived from synthetic patients

Methodology for generating and using digital twins for clinical outcome prediction. An approach that combines synthetic data and network science to create digital twins for precision medicine.

## Contents
- [synthetic_algorithms_comparison](https://github.com/EpistasisLab/SynTwin/tree/main/synthetic_algorithms_comparison) 
  - [step1_encoding_sampling ](https://github.com/EpistasisLab/SynTwin/blob/main/synthetic_algorithms_comparison/step1_encoding_sampling.ipynb)
  - [step2_synthetic_algorithms](https://github.com/EpistasisLab/SynTwin/tree/main/synthetic_algorithms_comparison/step2_synthetic_algorithms) 
  - [step3_synthetic_algorithms_comparision](https://github.com/EpistasisLab/SynTwin/blob/main/synthetic_algorithms_comparison/step3_synthetic_algorithms_comparision.ipynb)

- [SynTwin](https://github.com/EpistasisLab/SynTwin/tree/main/SynTwin)
  - [step1_data_cleaning_sampling](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step1_data_cleaning_sampling.ipynb)
  - [step2_mpom_synthetic_dataset](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step2_mpom_synthetic_dataset.ipynb)
  - [step3_data_preprocessing](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step3_data_preprocessing.ipynb)
  - [step4a_calc_distance_metrics_categorical](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step4a_calc_distance_matrices_categorical.ipynb)
  - [step4b_cdist_gower](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step4b_cdist_gower.py)
  - [step5a_percolation_threshold](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step5a_percolation_threshold.ipynb)
  - [step5b_percolation_threshold_calculation](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step5b_percolation_threshold_calculation.ipynb)
  - [step6a_get_resolution](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step6a_get_resolution.py)
  - [step6b_resolution_summarization](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step6b_resolution_summarization.ipynb)
  - [step7_vital_prediction](https://github.com/EpistasisLab/SynTwin/blob/main/SynTwin/step7_vital_prediction.ipynb)

We chose a population-based cancer registry from the Surveillance, Epidemiology, and End Results ([SEER](https://seer.cancer.gov)) program from the National Cancer Institute (USA) for this study due to its large sample size and ease of access by simple registration with an email address to allow for reproducibility. 

Follow the steps in [SynTwin](https://github.com/EpistasisLab/SynTwin/tree/main/SynTwin) to repeat the work from the paper. step2_mpom_synthetic_dataset can be replaced with any synthetic data generation algorithms that work best for your data. We evaluated three synthetic data generation algorithms, categorical latent Gaussian process (CLGP), mixture of product of multinomials (MPoM), and medical generative adversarial network (MC-MedGAN) by utilizing the code from [SYNDATA](https://github.com/LLNL/SYNDATA) and [multi-categorical-gans](https://github.com/rcamino/multi-categorical-gans). Please take a look at [synthetic_algorithms_comparison](https://github.com/EpistasisLab/SynTwin/tree/main/synthetic_algorithms_comparison) for details.

## Reference
Moore JH, Li X, Chang J-H, Tatonetti NP, Theodorescu D, Chen Y, Asselbergs F, Venkatesan M, Wang Z. Pacific Symposium on Biocomputing, in press (2024).
