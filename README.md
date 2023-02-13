# 490-poc
Examining the feasibility of quant methods in finance (a proof of concept).

## Models
Testing various RNN models to find strengths/weaknesses of each as it applies in micro & macroeconomic theory.
* LSTM: univariate, multivariate, and multiparallel

## Data
Data in csv format, preprocessed to calculate log returns and covariance.
* S&P 500: 100 largest companies by market cap, weekly intervals for 10 years
* VIX index to measure volatility

## Web App

Bootstrapped from the lightweight stack.
1. Backend: `source venv/bin/activate` && `poetry run uvicorn main:app --reload`
2. Frontend: `npm run dev`

**Backend:**
* FastAPI
* SQLite

**Frontend:**
* Vite, React, TypeScript
* Chakra UI
