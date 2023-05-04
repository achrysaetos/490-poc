# 490-poc
Examining the feasibility of quant methods in finance (a proof of concept).

## To run
*Initialize a python virtual environment and install requirements using pip (keras and transformers).*\
* LSTM - run `python testing.py` from the root directory
* BERT - `cd` into `sentiment/`, then run `python emotion.py`

## Models
Testing various RNN models to find strengths/weaknesses of each as it applies in micro & macroeconomic theory.
* LSTM: univariate (vanilla, stacked, bidirectional); multivariate; multiparallel
* NLP: BERT (Bidirectional Encoder Representations from Transformers)

## Data
Data in csv format, preprocessed to calculate log returns.
* S&P 500: weekly intervals for 20 years
* VIX index to measure volatility
