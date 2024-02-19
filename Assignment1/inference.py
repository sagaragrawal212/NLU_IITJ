from SentimentClassifier import SentimentClassifier

model = SentimentClassifier(model_name = 'decision_tree',
                            vectorizer_name = 'delta_tf_idf',
                           criterion = 'entropy')
data = ['this is a nice move.', 'sagar is good guy']

prediction = model.predict(pd.Series(data))

print(prediction)
