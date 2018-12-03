import numpy as np
import pandas as pd
import math
from numpy.linalg import inv

AVERAGE_REPUTAION = 1247
FEATURE_NUM = 7
LANGUAGE_POPULARITY = {"<javascript>": 0.698, "<html>": 0.685, "<css>": 0.651, "<sql>": 0.57, "<java>": 0.453, "<shell>": 0.398, "<bash>": 0.398, "<python>": 0.388, "<c#>": 0.344, "<php>":0.307, "<c++>": 0.254, "<c>": 0.23, "<typescript>": 0.174, "<ruby>": 0.101, "<swift>": 0.081, "<assembly>": 0.074, "<go>": 0.071, "<objective-c>": 0.07, "<vb.net>": 0.067, "<r>": 0.061, "<matlab>": 0.058, "<vba>": 0.049, "<kotlin>": 0.045, "<scala>": 0.044, "<groovy>": 0.043, "<perl>": 0.042, "no-tag": 0.01}

# features:
# language -- Zhiyuan
# userReputation -- Chong
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


def form_user_dict(user_reputation_data):
    userReputation = pd.read_csv(user_reputation_data)
    idList = list(userReputation["Id"])
    reputationList = list(userReputation["Reputation"])
    reputation_dict = {}
    for i in range(len(idList)):
        reputation_dict[idList[i]] = reputationList[i]

    return reputation_dict


def retrieve_valid_data(post_data, user_reputation_data):
    data = pd.read_csv(post_data)
    reputation_dict = form_user_dict(user_reputation_data)

    # the following lines are attribute from csv
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])
    tagList = list(data["Tags"])
    bodyList = list(data["Body"])
    postTypeList = list(data["PostTypeId"])
    titleList = list(data["Title"])
    userIdList = list(data["OwnerUserId"])

    #the following line is the features selected
    codeIncluded = list()
    bodyLength = list()
    postTypeID = list()
    titleLength = list()
    ownerUserReputation = list()
    languagePoularity = list()

    print("total posts: {}".format(len(viewCountList)))
    counter = 0
    for i in range(len(favoriteCountList)):
        if not math.isnan(favoriteCountList[i]) and not math.isnan(viewCountList[i]):
            counter += 1
            codeIncluded.append(is_code_included(bodyList[i]))
            bodyLength.append(count_body_length(bodyList[i]))
            postTypeID.append(postTypeList[i])
            titleLength.append(len(titleList[i]))
            reputaion = reputation_dict[userIdList[i]] if userIdList[i] in reputation_dict else 0
            ownerUserReputation.append(reputaion)
            languagePoularity.append(retrieve_tag_language_pop(str(tagList[i])))
            # if counter < 20:
            #     print(userIdList[i])
            #     print(reputation_dict[userIdList[i]])

            # if "c++" in tagList[i]:
            #     print(tagList[i])

    print("available posts: {}".format(counter))


def is_code_included(body):
    return "<code>" in body


def count_body_length(body):
    return len(body)


def retrieve_tag_language_pop(tag):
    if not tag:
        return LANGUAGE_POPULARITY["no-tag"]
    pop = 0
    for key in LANGUAGE_POPULARITY.keys():
        if key in tag:
            pop += LANGUAGE_POPULARITY[key]
    return pop if pop > 0 else LANGUAGE_POPULARITY["no-tag"]


def generate_user_reputation_query(input):
    data = pd.read_csv(input)
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])
    ownerUserIdList = list(data["OwnerUserId"])
    query = ""

    # "#standardSQL \n Select Id, Reputation from `sotorrent-org.2018_09_23.Users`"

    for i in range(len(favoriteCountList)):
        if not math.isnan(favoriteCountList[i]) and not math.isnan(viewCountList[i]) and not math.isnan(ownerUserIdList[i]):
            query = query + "or Id = " + str(int(ownerUserIdList[i])) + " "
    query = "where " + query[3:]
    print(query)

    return query


retrieve_valid_data("data.csv", "user_reputation_table.csv")

# generate_user_reputation_query("data.csv")
print(len(list(LANGUAGE_POPULARITY.keys())))



