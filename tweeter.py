import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter


if __name__ == '__main__':
    data = pd.read_csv('PrabowoGibran.csv', delimiter=';', encoding='utf-8')
    pd.set_option('display.max_columns', None)
    print(data.info())
    print(f'length of data : {len(data)}')

    data['full_text'] = data['full_text'].apply(
        lambda x: re.sub(r'@\w+', '', x)
    )

    data['hashtag'] = data['full_text'].apply(
        lambda x: re.findall(r'#\w+', x)
    )

    total_hashtag_counts = Counter()
    data['hashtag'].apply(total_hashtag_counts.update)
    most_used_hashtags = total_hashtag_counts.most_common(5)

    print(most_used_hashtags)
    print(data['full_text'].head(6))

    analyzer = SentimentIntensityAnalyzer()
    data['text_score'] = data['full_text'].apply(
        lambda x: analyzer.polarity_scores(x)
    )

    data['negative'] = data['text_score'].apply(lambda x: x['neg'] * 100)
    data['positive'] = data['text_score'].apply(lambda x: x['pos'] * 100)

    negative_sum = data['negative'].sum()
    positive_sum = data['positive'].sum()

    labels = ['Negative', 'Positive']
    sizes = [negative_sum, positive_sum]

    # Create a pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Sentiment Analysis of Prabowo's opinion on the 2024 presidential election")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
