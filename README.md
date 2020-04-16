<h1>Predicting payments in Swedish Motor Insurance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3753030.svg)](https://doi.org/10.5281/zenodo.3753030)

The experiment is based on Swedish Motor Insurance data set which describes third party automobile insurance claims for the year 1977 in Sweden. The goal here was to analyze the data and to perform simple linear regression for the 2 points of interest in this dataset: number of claims (the frequency) and sum of payments (the severity). In order to do that, we had to split the dataset into training and test set, apply predictions in the test set according to a training and to come up with measures for our predictions.

By visualizing our points of interest we get following distribution graphs:
![Distribution graphs](/output/plots/SwedishMotorInsurance_Distribution_Sibincic.png)

We can see that graphs are pretty similar, therefore we will try to do linear regression here based on number of claims to predict payments.

Later, we will measure how good our predictions are
