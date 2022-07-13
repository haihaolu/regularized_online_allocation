# Regularized Online Allocation
 
This repo consists python code for the paper "[Regularized online allocation problems: Fairness and beyond](https://arxiv.org/pdf/2007.00514.pdf)".

adx-alloc-data-2014 folder contains the data file generated following the procedure of "Yield Optimization of Display Advertising with Ad Exchange", Management Science. The dataset has 12 advertisers and 100,000 impressions. pub2-ads.txt contains the value of rho for each advertiser. pub2-sample.txt contains the revenue of matching each impression to the corresponding advertiser. We rescale the revenue so that the largest term is 1 in our experiment. We rescale \rho such that sum_j rho_j =1.5.

main.py is the main file to run the experiments, and save the output in csv files. Here are the documents for each optional arguments:

<pre>
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        The number of random trials.
  --lambd LAMBD         The coefficient of regularizer.
  --data_name DATA_NAME
                        The name of the dataset. Must be pub1-pub7.
  --step_size_constant STEP_SIZE_CONSTANT
                        The step-size constant.
  --regularizer REGULARIZER
                        The regularizer r.
  --reference REFERENCE
                        The reference function h.
  --save_frequency SAVE_FREQUENCY
                        How many iterations we save for the output.
  --T_ending T_ENDING   The number of samples.
  --num_T NUM_T         The number of T values.
  --sum_rho SUM_RHO     The sum of rho.
</pre>

### Dependency:
<pre>
cvxpy                              1.0.31
pandas                             0.20.2
numpy                              1.16.6
</pre>