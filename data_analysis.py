# import numpy as np
import pandas as pd
import math
from numpy.linalg import inv

# features: language, userReputation, codeIncluded, bodyLength

# def calculate_popularity(viewCount, ...):


def calculate_w(X, Y):
    # function to calculate the theta w
    # input should be numpy arrays
    return numpy.matmul(numpy.matmul(inv(numpy.matmul(X.transpose(), X)), X.transpose()), Y)


def retrieve_valid_data(input):
    data = pd.read_csv(input)
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])
    tagList = list(data["Tags"])

    print("total posts: {}".format(len(viewCountList)))
    counter = 0
    for i in range(len(favoriteCountList)):
        if not math.isnan(favoriteCountList[i]) and not math.isnan(viewCountList[i]):
            counter += 1
            if counter < 20:
                print(tagList[i])

    print("available posts: {}".format(counter))

calculate_valid_data("data.csv")


