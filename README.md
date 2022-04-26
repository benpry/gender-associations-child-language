# README

This repository provides the code for the following publication:

Prystawski, B., Grant, E., Nematzadeh, A., Lee, S.W.S., Stevenson, S., and Xu, Y. (to appear) The emergence of gender associations in child language development. Cognitive Science.

## Code

This folder contains all the code used to count word frequencies, create bootstrapped sub-samples of the corpora, compute correlations, and visualize the results. It is made up of the following sub-folders.

### read

This folder contains the code for reading the CHILDES and Switchboard corpora. Different files break the CHILDES corpus up by different variables, like the age of children or the decade of the conversations. There is no file for reading the Santa Barbara corpus since that is done directly in the bootstrapping file. The file `read_everything.sh` will run all of these files in parallel.

### bootstrap

This folder contains the code used to create bootstrapped instances of the Santa Barbara, Switchboard, and CHIDLES corpora. `create_bootstrapped_data.py` creates bootstrapped instances of the CHILDES corpus, while `bootstrap_santa_barbara.py` and `bootstrap_switchboard.py` create the bootstrapped instances of the Santa Barbara and Switchboard data. You can change the variable `n_iter` in each of these files to determine the number of bootstrapped instances to create. Beware that the default of 10,000 takes up a lot of time and disk space. `bootstrap_everything.sh` runs all of these files.

### analyze

This folder contains files which run the analysis. Each file contains code for one type of analysis. For instance `class_analysis.py` does the analysis by class. Most of the work in these analyses is done in `association_utils.py`, which contains functions that compute word embedding associations, vocabularies, and correlations. The results are output to subfolders of the `data/results` folder.

Some of these analysis files output text rather than saving csv files. `text_analyses.sh` runs all of these text-based analyses and stores the results in appropriately-named CSV files.

### visualize

The files in this folder create the figures and run the hypothesis tests for the paper. `MakeAllPlots.Rmd` is an R Markdown file that creats all the plots shown in this paper, with the exception of the tSNE-reduced word embedding plots whichare made in `analyze/tsne_plots.py`. `hypothesis_tests.py` runs all the relevant hypothesis tests and `run_hypothesis_tests.sh` runs the hypothesis tests and outputs them to a file in the results folder.

## Data

The subfolders here contain both data generated in the process of conducting the analyses or the final results of the analyses.

### raw_data

This folder contains pickle files with lists of all the words in the CHILDES corpus, broken down by gender, speech type. Some files are further broken down by class and decade. These pickle files are used as input to the bootstrapping process.

### bootstrapped_data

This folder contains the word-level statistics for the bootstrapped sub-samples of the CHILDES corpus. Including all sub-samples would make the number of bootstrapped instances extremely large, so we only included the first 10 of each type. 

### name_genders

This folder contains the lists of typically male and female names from the [Kantrowitz name corpus](https://www.cs.cmu.edu/Groups/AI/util/areas/nlp/corpora/names/), used to tag the names in the Santa Barbara corpus by gender.

### psycholing_data

This folder contains the lists of words by valence and concreteness, used in the analysis of psycholinguistic correlates of gender probability. 

### santa_barbara_data

This folder contains pickle files of all the words said by people with names identified as male and female in the Santa Barbara corpus. The pickle files are used in the bootstrapping process. The outputs of the bootstrapping are saved in the `bootstrapped` sub-folder.

### switchboard_data

This folder contains pickle files of all the words said by people tagged as male and female in the Switchboard corpus. The pickle files are used in the bootstrapping process. The outputs of the bootstrapping are saved in the `bootstrapped` sub-folder.

### whole_corpus

This contains aggregate word-level gender statistics computed using the whole corpus rather than any bootstrapped sub-samples. It is used in the psycholinguistic analysis. Concreteness data is available [here](http://crr.ugent.be/archives/1330) and valence data is available [here](http://crr.ugent.be/archives/1003).

### results

This is where the results of the analyses are stored. Correlations and mean gender associations are saved in the relevant sub-folder (e.g. `decade` for the analysis by decade). Each subfolder has two csv files, one of which ends in `nonames`. The `nonames` files contain the results of analysis that excludes names and explicitly-gendered words. There are also txt files reporting the results of hypothesis tests, linear regression, and psycholinguistic analyses. Finally, `similarity_correlation.csv` contains the correlation strengths between full-dimensional and dimensionality-reduced pairwise similarities for the words used in the t-SNE plots. Two subfolders are missing, `yearly` and `class`, corresponding to the yearly and class analysis respectively. This is because the files in these folders are too big for git. They can be downloaded from the OSF repository [here]().

## Demo

This folder contains a Jupyter notebook which demonstrates how gender probability is calculated. It is not used in the main analysis and is simply intended to help readers get a sense for how gender probability is calculated

# Figures

This folder contains the figures used in the main text and supplementary materials.
