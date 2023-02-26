from transformers import pipeline
import pandas as pd
import requests


def run_bert(string):
    classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
    prediction = classifier(string)
    return prediction[0][4]['score']


def main():
    file = "daily.csv" # location of data
    df = pd.read_csv(file)
    days = df["Date"].tolist() # one month back only, else api fails

    for i in range(len(days)):
        url = ('https://newsapi.org/v2/everything?'
            'q=%22s%26p%20500%22&'
            'searchIn=title&'
            f'from={days[i]}&'
            f'to={days[i]}&'
            'sortBy=popularity&'
            'language=en&'
            'apiKey=056bc424e60046f0a6c643dd2f69a2a1')
        response = requests.get(url)
        articles = response.json()["articles"]
        scores = []
        for article in articles:
            score = run_bert(article["title"])
            scores.append(score)
            print(days[i], article["title"], score)
        df.loc[i, "fear"] = sum(scores)/len(scores)
    df.to_csv(file, index=False)


if __name__ == "__main__":
    main()