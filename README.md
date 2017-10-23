# Titanic-Project
======================== Preprocessing ========================
The preprocessed data set is in the file Preprocessed/preprocessed.csv.
Since the missing values are mostly the ages of the passengers, they are the
only value being preprocessed and predicted.
The age of a passenger is predicted based on if the passenger survived.

Passengers are divided into several age groups: [0-9],[10,19], ..., [80-89]
After running the EM-Algorithm, the result shows that if the passenger survived,
s/he was mostly likely to be 30. Otherwise, s/he was most likely to be 20.

Below is the output of preprocess.py:
0: survived with prob 0.13103448275862067
10: didn't survive with prob 0.14386792452830188
20: didn't survive with prob 0.3372641509433963
30: survived with prob 0.2517241379310345
40: didn't survive with prob 0.1297169811320755
50: survived with prob 0.0689655172413793
60: didn't survive with prob 0.030660377358490573
70: didn't survive with prob 0.014150943396226417
80: survived with prob 0.003448275862068965
===================================================
If the target survived, his/her predicted age is 30
If the target died, his/her predicted age is 20

Explanation:
0: survived with prob 0.13103448275862067
  => this means that if this person was between 0 - 9, the probability of "this
    person survived" is greater than the probability of "this person didn't",
    and the value of the former is 0.13103448275862067.
