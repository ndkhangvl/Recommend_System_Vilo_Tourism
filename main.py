import random
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from pyvi import ViTokenizer
import seaborn as sns
import pandas as pd
import numpy as np
from zipfile import ZipFile
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import json


def content_based_recommendation_vietnamese(data, user_hashtags, num_recommendations=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    user_hashtags_str = " ".join(user_hashtags)
    user_hashtags_vector = vectorizer.transform([user_hashtags_str])
    similarity_scores = cosine_similarity(user_hashtags_vector, tfidf_matrix)
    recommended_indices = similarity_scores.argsort()[0][-num_recommendations:][::-1]
    recommended_objects = [data[idx] for idx in recommended_indices]
    return recommended_objects


loaded_model = keras.models.load_model("recommend_tourism")

info_tourism = pd.DataFrame()
tourism_rating = pd.DataFrame()
users = pd.DataFrame()


app = Flask(__name__)


@app.route("/", methods={"GET"})
def home():
    return "Xin chào tôi là ndkhangvl"


@app.route("/test", methods=["GET"])
def test():
    response = requests.get("http://127.0.0.1:8000/place")

    if response.status_code == 200:
        # Lấy dữ liệu từ phản hồi của API
        data = response.json()
        names = [item["describe_place"] for item in data]
    return jsonify({"recommendations": names})


@app.route("/recommend_tourism", methods=["POST"])
def recommend_tourism():
    try:
        # loaded_model = keras.models.load_model("recommend_tourism")
        # print(loaded_model.summary())
        # response_tour = requests.get("http://127.0.0.1:8000/api/recommend-place")
        # response_rating = requests.get("http://127.0.0.1:8000/api/recommend-rating")
        # response_user = requests.get("http://127.0.0.1:8000/api/recommend-user")

        # if (
        #     response_tour.status_code == 200
        #     and response_rating.status_code == 200
        #     and response_user.status_code == 200
        # ):
        #     tour_data = response_tour.json()
        #     rating_data = response_rating.json()
        #     user_data = response_user.json()

        #     info_tourism = pd.DataFrame(tour_data)
        #     tourism_rating = pd.DataFrame(rating_data)
        #     users = pd.DataFrame(user_data)

        # else:
        #     print("Failed to retrieve JSON data from the API.")

        global info_tourism, tourism_rating, users
        if info_tourism.empty or tourism_rating.empty or users.empty:
            # If dataframes are empty, fetch data from APIs
            info_tourism, tourism_rating, users = fetch_data_from_apis()

        # info_tourism = pd.read_csv(f"./vilo_tour.csv")
        # tourism_rating = pd.read_csv(f"./rating_final.csv")
        # users = pd.read_csv(f"./vilo_user.csv")

        # user_id = request.json.get("id_user")
        user_id = str(request.json.get("id_user"))

        tourism_all = np.concatenate(
            (info_tourism.id_place.unique(), tourism_rating.id_place.unique())
        )

        tourism_all = np.sort(np.unique(tourism_all))
        df = tourism_rating
        # print(df)
        user_ids = df.id_user.unique().tolist()

        user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

        user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

        id_places = df.id_place.unique().tolist()

        place_to_place_encoded = {x: i for i, x in enumerate(id_places)}

        place_encoded_to_place = {x: i for x, i in enumerate(id_places)}

        df["user"] = df.id_user.map(user_to_user_encoded)

        df["place"] = df.id_place.map(place_to_place_encoded)

        all_tourism_rate = tourism_rating

        all_tourism = pd.merge(
            all_tourism_rate,
            info_tourism[
                [
                    "id_place",
                    "name_place",
                    "address_place",
                    "phone_place",
                    "email_contact_place",
                    "image_url",
                ]
            ],
            on="id_place",
            how="left",
        )

        preparation = all_tourism.drop_duplicates("id_place")

        id_place = preparation.id_place.tolist()

        place_name = preparation.name_place.tolist()

        place_category = preparation.address_place.tolist()

        place_desc = preparation.phone_place.tolist()

        place_city = preparation.email_contact_place.tolist()

        place_image = preparation.image_url.tolist()

        tourism_new = pd.DataFrame(
            {
                "id": id_place,
                "name": place_name,
                "address": place_category,
                "phone": place_desc,
                "email": place_city,
                "image_url": place_image,
                # "city_category":city_category
            }
        )

        place_df = tourism_new
        # df = pd.read_csv(f"./rating_final.csv")
        print("Test dữ liệu")
        print(place_df)
        place_visited_by_user = df[df.id_user == user_id]

        if len(place_visited_by_user) == 0:
            random_recommendations = place_df.sample(n=5)
            random_recommendations_json = random_recommendations.to_dict(
                orient="records"
            )

            # json_data = (
            #     json.dumps(random_recommendations_json, ensure_ascii=False)
            #     .encode("utf-8")
            #     .decode()
            # )
            return jsonify(random_recommendations_json)

        place_not_visited = place_df[
            ~place_df["id"].isin(place_visited_by_user["id_place"].values)
        ]["id"]

        place_not_visited = list(
            set(place_not_visited).intersection(set(place_to_place_encoded.keys()))
        )

        place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]

        print(len(place_not_visited))

        if len(place_not_visited) == 0:
            random_recommendations = place_df.sample(n=5)
            random_recommendations_json = random_recommendations.to_dict(
                orient="records"
            )

            json_data = (
                json.dumps(random_recommendations_json, ensure_ascii=False)
                .encode("utf-8")
                .decode()
            )
            return jsonify({"random_recommendations": json_data})

        user_encoder = user_to_user_encoded.get(user_id)
        user_place_array = np.hstack(
            ([[user_encoder]] * len(place_not_visited), place_not_visited)
        ).astype(np.int64)
        # user_place_array = np.hstack(
        #     [[np.int64(user_encoder)] * len(place_not_visited), place_not_visited]
        # )

        ratings = loaded_model.predict(user_place_array).flatten()

        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_id_places = [
            place_encoded_to_place.get(place_not_visited[x][0])
            for x in top_ratings_indices
        ]

        print("Người dùng với lượt đánh giá cao")

        top_place_user = (
            place_visited_by_user.sort_values(by="place_ratings", ascending=False)
            .head(5)
            .id_place.values
        )

        # place_df_rows = place_df[place_df["id"].isin(top_place_user)]
        recommended_place = place_df[place_df["id"].isin(recommended_id_places)]
        # print(recommended_place)
        # Convert the DataFrame to a list of dictionaries
        data_as_dict = recommended_place.to_dict(orient="records")
        # print("Test")
        # print(data_as_dict)

        return jsonify(data_as_dict)
    except Exception as e:
        print(f"Lỗi: {str(e)}")


def fetch_data_from_apis():
    try:
        response_tour = requests.get("http://127.0.0.1:8000/api/recommend-place")
        response_rating = requests.get("http://127.0.0.1:8000/api/recommend-rating")
        response_user = requests.get("http://127.0.0.1:8000/api/recommend-user")

        if (
            response_tour.status_code == 200
            and response_rating.status_code == 200
            and response_user.status_code == 200
        ):
            tour_data = response_tour.json()
            rating_data = response_rating.json()
            user_data = response_user.json()

            info_tourism = pd.DataFrame(tour_data)
            tourism_rating = pd.DataFrame(rating_data)
            users = pd.DataFrame(user_data)

            return info_tourism, tourism_rating, users
        else:
            app.logger.error("Failed to retrieve JSON data from the API.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        app.logger.error(f"Error fetching data from APIs: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@app.route("/recommend", methods=["POST"])
def recommend():
    response = requests.get("http://127.0.0.1:8000/place")

    if response.status_code == 200:
        data = response.json()
        names = [item["name_place"] for item in data]

    user_hashtags = request.json["hashtags"]
    split_user_hashtags = [hashtag.split("_") for hashtag in user_hashtags]
    tokenized_user_hashtags = [
        word for hashtag in split_user_hashtags for word in hashtag
    ]
    recommendations = content_based_recommendation_vietnamese(
        names, tokenized_user_hashtags
    )
    return jsonify({"recommendations": recommendations})


@app.route("/recommend_content", methods=["POST"])
def recommend_content():
    query = request.json["query"]

    api_url = "http://127.0.0.1:8000/place"
    response = requests.get(api_url)
    data = response.json()

    # Tạo ma trận TF-IDF cho các mô tả địa điểm
    tfidf_vectorizer = TfidfVectorizer()
    descriptions = [item["describe_place"] for item in data]
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

    # Tính độ tương đồng giữa câu truy vấn và các mô tả địa điểm
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sắp xếp và trả về các đề xuất dựa trên độ tương đồng
    indices = similarities.argsort()[::-1]
    recommendations = [data[idx]["name_place"] for idx in indices][:3]

    return jsonify(recommendations)


if __name__ == "__main__":
    app.run()
