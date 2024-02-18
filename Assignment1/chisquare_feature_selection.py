from sklearn.feature_selection import chi2

def chi_square_tfidf_feature_selection(X_train_tfidf, y_train, k=1000):
  """
  Performs feature selection using Chi-square and TF-IDF.
  """

  # Perform Chi-square test
  chi2_scores, _ = chi2(X_train_tfidf, y_train)

  # Get indices of top k features
  feature_indices = chi2_scores.argsort()[-k:]

  return feature_indices
