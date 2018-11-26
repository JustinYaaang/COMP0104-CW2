import numpy as np
import pandas as pd
import math
from numpy.linalg import inv

# features:
# language
# userReputation
# codeIncluded -- done
# bodyLength  -- done
# postTypeID  -- done (could be re-consider)
# titleLength -- done

# def calculate_popularity(viewCount, ...):


def calculate_w(X, Y):
    # function to calculate the theta w
    # input should be numpy arrays
    return np.matmul(np.matmul(inv(np.matmul(X.transpose(), X)), X.transpose()), Y)


def perdict_y(x, w):
    return np.matmul(w, x)


def retrieve_valid_data(input):
    data = pd.read_csv(input)
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])
    tagList = list(data["Tags"])
    bodyList = list(data["Body"])
    postTypeList = list(data["PostTypeId"])
    titleList = list(data["Title"])
    codeIncluded, bodyLength, postTypeID, titleLength = list(), list(), list(), list()

    print("total posts: {}".format(len(viewCountList)))
    counter = 0
    for i in range(len(favoriteCountList)):
        if not math.isnan(favoriteCountList[i]) and not math.isnan(viewCountList[i]):
            counter += 1
            codeIncluded.append(is_code_included(bodyList[i]))
            bodyLength.append(count_body_length(bodyList[i]))
            postTypeID.append(postTypeList[i])
            titleLength.append(len(titleList[i]))

            # if counter < 20:
            #     print(tagList[i])

    print("available posts: {}".format(counter))


def is_code_included(body):
    return "<code>" in body


def count_body_length(body):
    return len(body)


retrieve_valid_data("data.csv")

