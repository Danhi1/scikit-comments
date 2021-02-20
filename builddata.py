import json
import random
import numpy as np

# Anything that's not a word (numbers, special symbols, punctionation, links, emoji)
# Will be detrimental to the classifier as learning data hence the filter
DEFAULT_WHITELIST = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ'
JSON = './data/reviews/Video_Games_5.json' # Location of reviews file
SAVEPATH = "./data/reviews" # Savepath for balanced and filtered data


# Enum
class Opinions:
    NEG = 'NEGATIVE'
    NTL = 'NEUTRAL'
    POS = 'POSITIVE'

class Review:
    def __init__(self, text, rating):
        # Review text
        self.text = text
        # Review rating: 0 to 5 stars
        self.rating = rating
        # General opinion on the book (positive, neutral, negative)
        self.opinion = self.get_opinion()
        
    def get_opinion(self):
        # 0 to 2 stars
        if self.rating <= 2:
            return Opinions.NEG
        # 3 stars
        elif self.rating == 3:
            return Opinions.NTL
        # 4 to 5 stars
        else:                  
            return Opinions.POS
        



class Filter:
    def __init__(self, whitelist = set(DEFAULT_WHITELIST)):
        self.whitelist = whitelist
    
    def filter_data(self, text):
        filtered_text = ''.join(filter(self.whitelist.__contains__, text))
        return filtered_text


class Balancer():
    @staticmethod
    
    def balance(data, max_length = 5000): # Consider lowering it on weaker machines
        positive = list(filter(lambda x: x.opinion == Opinions.POS, data))
        neutral = list(filter(lambda x: x.opinion == Opinions.NTL, data))
        negative = list(filter(lambda x: x.opinion == Opinions.NEG, data))

        data_lenght = min(len(positive), len(neutral), len(negative), max_length)
        balanced_data = positive[:data_lenght] + neutral[:data_lenght] + negative[:data_lenght]
        random.shuffle(balanced_data)
        return balanced_data

    def check_balance(data):
        positive = list(filter(lambda x: x.opinion == Opinions.POS, data))
        neutral = list(filter(lambda x: x.opinion == Opinions.NTL, data))
        negative = list(filter(lambda x: x.opinion == Opinions.NEG, data))

        return len(positive), len(neutral), len(negative)

reviews = []
symbol_filter = Filter()

print("Loading and filtering the reviews...")
with open(JSON) as f:
    for line in f:
        review = json.loads(line)
        
        text = symbol_filter.filter_data(review['reviewText'])
        rating = review['overall']
        
        reviews.append(Review(text, rating))
        

print("TOTAL REVIEWS:")
print("POSITIVE, NEUTRAL, NEGATIVE:", Balancer.check_balance(reviews), "\n...")

print("BALANCED REVIEWS:")
reviews_balanced = Balancer.balance(reviews)
print("POSITIVE, NEUTRAL, NEGATIVE:", Balancer.check_balance(reviews_balanced), "\n...")

np.save(SAVEPATH, reviews_balanced)
print("DATA SAVED AS", SAVEPATH + ".npy")
