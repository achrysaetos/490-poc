# 490-poc
Examining the feasibility of quant methods in finance (a proof of concept).

## Models
Testing various RNN models to find strengths/weaknesses of each as it applies in micro & macroeconomic theory.
* LSTM: univariate (vanilla, stacked, bidirectional); multivariate; multiparallel
* NLP: Bert (Bidirectional Encoder Representations from Transformers)

## Data
Data in csv format, preprocessed to calculate log returns.
* S&P 500: weekly intervals for 20 years
* VIX index to measure volatility

## Web App
Not functional (needed?) yet.
1. Backend (FastAPI, SQLite): `source venv/bin/activate` && `poetry run uvicorn main:app --reload`
2. Frontend (Vite, React, TypeScript, Chakra UI): `npm run dev`
