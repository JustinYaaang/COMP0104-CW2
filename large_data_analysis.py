import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()

AVERAGE_REPUTAION = 1247
FEATURE_NUM = 8
LANGUAGE_POPULARITY = {"<javascript>": 0.698, "<html>": 0.685, "<css>": 0.651, "<sql>": 0.57, "<java>": 0.453,
                       "<shell>": 0.398, "<bash>": 0.398, "<python>": 0.388, "<c#>": 0.344, "<php>": 0.307,
                       "<c++>": 0.254, "<c>": 0.23, "<typescript>": 0.174, "<ruby>": 0.101, "<swift>": 0.081,
                       "<assembly>": 0.074, "<go>": 0.071, "<objective-c>": 0.07, "<vb.net>": 0.067, "<r>": 0.061,
                       "<matlab>": 0.058, "<vba>": 0.049, "<kotlin>": 0.045, "<scala>": 0.044, "<groovy>": 0.043,
                       "<perl>": 0.042, "no-tag": 0.01}
BOUNDARY = 200
MISSING_IDS = [49448198 , 49366664 , 49239691 , 49442315 , 49426445 , 49283726 , 49409679 , 49271952 , 49371287 , 49377562 , 49349147 , 49305117 , 49338525 , 49273502 , 49374239 , 49444897 , 49314209 , 49252259 , 49210275 , 49347230 , 49360167 , 49249964 , 49311277 , 49395374 , 49356848 , 49301939 , 49425719 , 49262136 , 49462711 , 49386942 , 49247679 , 49252288 , 49430082 , 49270850 , 49383364 , 49275204 , 49346500 , 49361223 , 49385672 , 49338824 , 49335114 , 49303882 , 49356364 , 49311050 , 49362254 , 49300558 , 49298517 , 49332823 , 49426392 , 49298812 , 49296986 , 49267930 , 49255644 , 49273946 , 49221982 , 49272798 , 49212633 , 49274465 , 49404131 , 49312102 , 49366887 , 49322601 , 49342186 , 49383919 , 49428720 , 49395698 , 49212146 , 49399925 , 49394294 , 49412729 , 49333499 , 49252732]
MISSING_ID_DICT = {}

# features:

# languagePoularity -- Zhiyuan
# ownerUserReputation -- Chong
# codeIncluded -- done
# bodyLength  -- done
# titleLength -- done
# oldViewCount
# oldFavoriteCount

def generate_missing_id_dict():
    for ele in MISSING_IDS:
        MISSING_ID_DICT[ele] = 1


def calculate_popularity(viewCount, favoriteCount):
    pop_list = []
    for i in range(len(viewCount)):
        view, favorite = str(viewCount[i]), str(favoriteCount[i])
        pop_list.append(0.9 * int(view) + 0.1 * int(favorite))
    return pop_list
    # return list(map(lambda x, y: 0.9 * int(x) + 0.1 * int(y), viewCount, favoriteCount))


def calculate_theta(X, Y):
    # function to calculate the theta
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


def retrieve_latest_data(latest_data_sorted):
    data = pd.read_csv(latest_data_sorted)
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])

    for i in range(len(viewCountList)):
        viewCountList[i] = int(viewCountList[i])
        favoriteCountList[i] = int(favoriteCountList[i])
    return viewCountList, favoriteCountList


def retrieve_valid_data(post_data, user_reputation_data, latest_data_sorted):

    data = pd.read_csv(post_data)
    reputation_dict = form_user_dict(user_reputation_data)

    # the following lines are attribute from csv
    idList = list(data['Id'])
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])
    tagList = list(data["Tags"])
    bodyList = list(data["Body"])
    postTypeList = list(data["PostTypeId"])
    titleList = list(data["Title"])
    userIdList = list(data["OwnerUserId"])

    # the following line is the features selected for X
    codeIncluded = list()
    bodyLength = list()
    postTypeID = list()
    titleLength = list()
    ownerUserReputation = list()
    languagePoularity = list()

    # the following line are features to combine
    viewCount = list()
    favoriteCount = list()

    print("total posts: {}".format(len(viewCountList)))
    counter = 0
    for i in range(len(favoriteCountList)):
        if idList[i] in MISSING_ID_DICT:
            continue
        counter += 1

        # for X
        codeIncluded.append(is_code_included(str(bodyList[i])))
        bodyLength.append(count_body_length(bodyList[i]))
        postTypeID.append(postTypeList[i])
        titleLength.append(count_body_length(titleList[i]))
        reputaion = reputation_dict[userIdList[i]] if userIdList[i] in reputation_dict else 0
        ownerUserReputation.append(reputaion)
        languagePoularity.append(retrieve_tag_language_pop(str(tagList[i])))

        # for Y
        viewCount.append(viewCountList[i])
        favoriteCount.append(favoriteCountList[i])
        # if counter == 5:
        #     break
        # if counter < 20:
        #     print(userIdList[i])
        #     print(reputation_dict[userIdList[i]])

        # if "c++" in tagList[i]:
        #     print(tagList[i])

    latest_view_count, latest_favorite_count = retrieve_latest_data(latest_data_sorted)
    popularityCount = calculate_popularity(latest_view_count, latest_favorite_count)
    print("latest available posts: {}".format(len(latest_view_count)))

    cutoff = int(counter * 0.8)
    training_data_X = np.zeros(shape=(cutoff, FEATURE_NUM))
    test_data_X = np.zeros(shape=(int(counter - cutoff), FEATURE_NUM))
    training_data_Y = np.zeros(shape=(cutoff, 1))
    test_data_Y = np.zeros(shape=(int(counter - cutoff), 1))

    for i in range(counter):
        if i < cutoff:
            # training_data_X[i] = [1, languagePoularity[i], ownerUserReputation[i], codeIncluded[i], bodyLength[i], postTypeID[i], titleLength[i]]
            training_data_X[i] = [1, languagePoularity[i], ownerUserReputation[i], codeIncluded[i], bodyLength[i], titleLength[i], int(viewCount[i]), int(favoriteCount[i])]
        else:
            # test_data_X[i - cutoff] = [1, languagePoularity[i], ownerUserReputation[i], codeIncluded[i], bodyLength[i], postTypeID[i], titleLength[i]]
            test_data_X[i - cutoff] = [1, languagePoularity[i], ownerUserReputation[i], codeIncluded[i], bodyLength[i], titleLength[i], int(viewCount[i]), int(favoriteCount[i])]

    training_data_Y = popularityCount[:cutoff]
    test_data_Y = popularityCount[cutoff:]
    print("available posts: {}".format(counter))
    return training_data_X, test_data_X, training_data_Y, test_data_Y


def is_code_included(body):
    return "<code>" in body


def count_body_length(body):
    if type(body) == type("aa"):
        return len(body)
    else:
        return 0


def count_title_length(title):
    return len(title)


def retrieve_tag_language_pop(tag):
    if not tag:
        return LANGUAGE_POPULARITY["no-tag"]
    pop = 0
    for key in LANGUAGE_POPULARITY.keys():
        if key in tag:
            pop += LANGUAGE_POPULARITY[key]
    return pop if pop > 0 else LANGUAGE_POPULARITY["no-tag"]


def generate_latest_data(input):
    data = pd.read_csv(input)
    post_id = list(data['Id'])
    file = open("latest_data.txt", "w")
    query = ""
    for i in range(len(post_id)):
        query += " or Id = " + str(post_id[i])
    file.write(query)
    file.close()


def generate_user_reputation_query(input):

    data = pd.read_csv(input)
    viewCountList = list(data['ViewCount'])
    favoriteCountList = list(data["FavoriteCount"])
    ownerUserIdList = list(data["OwnerUserId"])
    query = ""

    # "#standardSQL \n Select Id, Reputation from `sotorrent-org.2018_09_23.Users`"

    for i in range(len(favoriteCountList)):
        if not math.isnan(ownerUserIdList[i]):
            query = query + "or Id = " + str(int(ownerUserIdList[i])) + " "
    query = "where " + query[3:]
    print(query)
    file = open("userID.txt", "w")
    file.write(query)
    file.close()

    return query


def main(data_file, user_reputation_file, latest_data_sorted):
    generate_missing_id_dict()
    training_data_X, test_data_X, training_data_Y, test_data_Y = retrieve_valid_data(data_file, user_reputation_file, latest_data_sorted)
    model.fit(training_data_X, training_data_Y)
    # predicted_training_Y = list(model.predict(training_data_X))

    predicted_test_Y = list(model.predict(test_data_X))
    # print("test_data_Y is:", test_data_Y)
    # print("-------------------------------------------")
    # print("predicted_test_Y is:", predicted_test_Y)
    R_Square = r2_score(test_data_Y, predicted_test_Y)
    print(model.coef_)
    print(model.intercept_)
    print("R_Square is ", R_Square)


def find_diff(input1, input2):
    counter = 0
    data_1 = pd.read_csv(input1)
    post_id_1 = list(data_1['Id'])

    data_2 = pd.read_csv(input2)
    post_id_2 = list(data_2['Id'])

    difference = list(set(post_id_1) - set(post_id_2))
    query = ""
    for i in range(len(difference)):
        counter += 1
        query = query + ", " + str(int(difference[i])) + " "

    print(counter)
    file = open("missingPostID.txt", "w")
    file.write(query)
    file.close()

# generate_user_reputation_query("10000/1.csv")
main("10000/1_sorted.csv", "10000/userReputaion.csv", "10000/latest_data_sorted.csv")

# generate_latest_data("10000/1.csv")
# find_diff("10000/1_sorted.csv", "10000/latest_data_sorted.csv")







