import pickle 

with open("./Models/scikit-comments.pkl", "rb") as f:
    classifier = pickle.load(f)
    vectorizer = pickle.load(f)


final_test = ['this is a very good videogame and I enjoyed it a lot',
              'average and mediocre',
              'terrible game do not buy']

final_test_vectors = vectorizer.transform(final_test)

print(classifier.predict(final_test_vectors))
