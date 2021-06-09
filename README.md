Code: Operationalizing Complex Causes: A Pragmatic View of Mediation
====================================================================

This folder contains code to reproduce results for the ICML 2021 paper. Please note that certain files referenced by some of the scripts included.

## Prereqs

The following will need to be downloaded and generated for the code to run:

#### Python_Img_Humor/data/
1. you will need glove.6B, downloaded from https://nlp.stanford.edu/projects/glove/.
2. (optional) one could also use GoogleNews-vectors-negative300.bin.gz, download from https://code.google.com/archive/p/word2vec/, under 'Pre-trained word and phrase vectors'.
3. you will need semeval-2020-task-7-dataset/\*, downloaded from https://www.cs.rochester.edu/u/nhossain/humicroedit.html, 'Full dataset release'.

#### Python_Img_Humor/data/ImgPretSim/diffY and Python_Img_Humor/data/Humicroedit/diffY
To reproduce Figure 8 you will need to generate the datasets with different Y configs, by running the last cell in the notebooks Python_Img_Humor/data_generation_notebooks/ImgPretSim/ImgPretSim.ipynb and Python_Img_Humor/data_generation_notebooks/Humicroedit/Humicroedit.ipynb.


## Image Perturbation
Code can be found in Python_Img_Humor/src and data can be found in Python_Img_Humor/data/ImgPertSim. Description of dataset construction is in the Jupyter notebook Python_Img_Humor/data_generation_notebooks/ImgPertSim/ImgPertSim.ipynb. The code is written in Pytorch. To reproduce results follow:

1. Make sure requirements are installed, i.e. pip install requirements.txt (suggested: use GPU if available). Alternatively, use the environment.yml file, and create a virtual environment with 

```
conda env create -f environment.yml
conda activate ComplexCauses
```

2. Configurations for the Image Preturbation setup are indicated in Python_Img_Humor/src/configs/causal_effect_config.yml. Make sure the ones under "ImgPertSim" headlines are uncommented (and make sure to comment out Humicroedit configs).
3. Call the command "python main.py"
4. Results will appear in Python_Img_Humor/out/viz/ImgPertSim. MSE_vs_Y_labels_perc.png will correspond to Figure 5. 
5. To reproduce Figure 6, first consider the parameters of fitted model, saved in df_params_PertImgSim.csv. Next, call the conditional independence test in R_cond_ind_test/ci_test_img.R on the files Python_Img_Humor/out/viz/ImgPertSim/residuals_nested=True_PertImgSim.csv and Python_Img_Humor/out/viz/ImgPertSim/residuals_nested=False_PertImgSim.csv. nested=True matches the model \Phi ~ Z, and nested=False matches the model \Phi ~ W,Z. Results will be saved in R_cond_ind_test/ci_test_results_img.csv

## Humor edits
Code can be found in Python_Img_Humor/src and data can be found in Python_Img_Humor/data/Humicroedit. Description of dataset construction is in the Jupyter notebook Python_Img_Humor/data_generation_notebooks/Humicroedit/Humicroedit.ipynb. The code is written in Pytorch. To reproduce results follow:

1. Make sure requirements are installed, i.e. pip install requirements.txt (suggested: use GPU if available)
2. Configurations for the Image Preturbation setup are indicated in Python_Img_Humor/src/configs/causal_effect_config.yml. Make sure the ones under "Humicroedit" headlines are uncommented (and make sure to comment out ImgPert configs).
3. Call the command "python main.py"
4. Results will appear in Python_Img_Humor/out/viz/Humicroedit. MSE_vs_Y_labels_perc.png will correspond to Figure 5. 
5. To reproduce Figure 6, first consider the parameters of fitted model, saved in df_params_Humicroedit.csv. Next, call the conditional independence test in R_cond_ind_test/ci_test_humor.R . The paths to the correct files is already prespecified in the file, but for completeness, it will use the files Python_Img_Humor/out/viz/Humicroedit/residuals_nested=True_PertImgSim.csv and Python_Img_Humor/out/viz/Humicroedit/residuals_nested=False_Humicroedit.csv. nested=True matches the model \Phi ~ Z, and nested=False matches the model \Phi ~ W,Z. Results will be saved in R_cond_ind_test/ci_test_results_humor.csv

## Gene Knockouts
Code can be found in R_genomics/. The data was too big to include herein, but instructions for download are included. To reproduce results follow:

1. Your working directory will need the following (already created in this submission) directories ready, where model objects and simulated data will be stored: "genie3_models" (this will contain model objects corresponding to the 4177 structural equations of the GENIE3 model); "simulations" (this will contain data for Z, W, X, Phi, Phi_hat, and Y); and "phi_models" (this will contain model objects for the 168 E[\Phi | Z, W] regressions).
2. Download net3_expression_data.tsv from http://dreamchallenges.org/project/dream-5-network-inference-challenge/ and place it in your working directory.
3. Call "Rscript pkgs.R" to install all requisite libraries.
4. Call "Rscript fit_genie3.R" to train GENIE3 on the E. Coli dataset and export model objects (this takes ~8 hrs).
5. Call "Rscript e_coli_sim.R" to simulate Z, X, and Phi (this takes ~1 hr).
6. Call "Rscript phi_models.R" to simulate Phi_hat (this takes ~4 hrs).
7. Call "Rscript y_model.R" to simulate Y and fit a lasso model for E[Y | Phi_hat]
8. Call "Rscript baselines.R" to fit a series of baseline models for E[Y | Z, W] with increasing amounts of data.
9. Call "Rscript discover_m.R" to perform association tests for Phi and W, thereby completing the mediator discovery pipeline.
