1. For each datasets, we randomly split it into three parts, i.e., 60% for training, 20% for validation and the remaining 20% for test. We take the top 10% and 20% most popular items as head items and the rest as the tail ones. In the validation data, we further obtain a tail data by removing the top 10% most popular items, which is used for parameter tuning. Similarly, in the test data, we obtain two tail data by removing the top 10% or 20% most popular items.

2. We select the best combination of parameters by using "python3 tune_parameters.py"

3. We report the best combination of parameters in "run.sh", and check the performance on test data by using "./run.sh" on the Linux platform.

4. We tune the parameters and check the performance of all methods on an Ubuntu16 machine with 8 P100. At the same time, we find that the results seem not to be the same under windows platform due to some reasons like floating-point accuracy, so we recommend running this code on the Linux platform.
