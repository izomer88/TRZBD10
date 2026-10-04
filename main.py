import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


# ============================================================
# 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
# ============================================================

print("\nЗагружаем и подготавливаем данные...")

positive_reviews = [
    "отлично",
    "замечательно",
    "супер",
    "рекомендую",
    "лучший",
    "качество отличное",
    "доволен",
    "прекрасно",
    "великолепно",
    "нравится",
]

negative_reviews = [
    "ужасно",
    "плохо",
    "разочарован",
    "не рекомендую",
    "брак",
    "качество плохое",
    "ужасное качество",
    "не работает",
    "деньги на ветер",
    "кошмар",
]

products = ["Смартфон", "Ноутбук", "Наушники", "Часы", "Планшет"]

data = []

for i in range(200):
    rating = np.random.choice(
        [1, 2, 3, 4, 5],
        p=[0.1, 0.1, 0.2, 0.3, 0.3]
    )

    if rating >= 4:
        text = np.random.choice(positive_reviews)
        sentiment = 1
    elif rating <= 2:
        text = np.random.choice(negative_reviews)
        sentiment = 0
    else:
        text = "нормальный товар"
        sentiment = None

    data.append({
        "id": i,
        "text": text,
        "rating": rating,
        "sentiment": sentiment,
        "date": datetime.now() - timedelta(days=np.random.randint(0, 60)),
        "product": np.random.choice(products),
        "user_id": np.random.randint(1, 31),
        "user_name": f"Клиент_{np.random.randint(1, 31)}",
        "phone": f"+7{np.random.randint(900, 999)}{np.random.randint(1000000, 9999999)}",
    })

df = pd.DataFrame(data)
print(f"Создано {len(df)} записей")


# ============================================================
# 2. КЛАССИФИКАЦИЯ ТЕКСТОВ
# ============================================================

print("\nКЛАССИФИКАЦИЯ ТЕКСТОВ")

df_binary = df[df["sentiment"].notna()].copy()

X = df_binary["text"].values
y = df_binary["sentiment"].values.astype(int)

print(
    f"Обучающих примеров: {len(X)} "
    f"(позитив: {sum(y)}, негатив: {len(y) - sum(y)})"
)

vectorizer = TfidfVectorizer(max_features=100)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\nМЕТРИКИ:")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")

print("\nМатрица ошибок:")
print(
    f"Негатив → предсказано негатив: {cm[0, 0]}, "
    f"позитив: {cm[0, 1]}"
)
print(
    f"Позитив → предсказано негатив: {cm[1, 0]}, "
    f"позитив: {cm[1, 1]}"
)


# ============================================================
# тональност и процент уверенности
# ============================================================

def analyze_sentiment(text):

    x_new = vectorizer.transform([text])

    prediction = int(model.predict(x_new)[0])
    probabilities = model.predict_proba(x_new)[0]

    confidence = probabilities[prediction] * 100
    sentiment_name = "ПОЗИТИВНЫЙ" if prediction == 1 else "НЕГАТИВНЫЙ"

    return sentiment_name, confidence


print("\nПРОВЕРКА НА НОВЫХ ПРИМЕРАХ:")

test_phrases = [
    "отличный товар, всем рекомендую",
    "ужасное качество, разочарован",
    "нормально, но ничего особенного",
]

for phrase in test_phrases:
    sentiment_name, confidence = analyze_sentiment(phrase)
    print(
        f'"{phrase}" → {sentiment_name} '
        f"(уверенность: {confidence:.1f}%)"
    )


# ============================================================
# Топ товаров по доле положительных отзывов
# ============================================================

def top_products_by_positive(dataframe):
    """
    Возвращает таблицу товаров, отсортированную
    по доле положительных отзывов.
    """
    product_stats = (
        dataframe[dataframe["sentiment"].notna()]
        .groupby("product")["sentiment"]
        .agg(["count", "sum"])
    )

    product_stats["positive_percent"] = (
        product_stats["sum"] / product_stats["count"] * 100
    )

    return product_stats.sort_values(
        "positive_percent",
        ascending=False
    )


top_products = top_products_by_positive(df)

print("\nТОП ТОВАРОВ ПО ДОЛЕ ПОЗИТИВА:")

for product_name, row in top_products.iterrows():
    print(
        f"{product_name}: {row['positive_percent']:.2f}% "
        f"(позитивных: {int(row['sum'])}, всего: {int(row['count'])})"
    )


# ============================================================

# 5 самых частых слов в негативных отзывах
# ============================================================

def top_negative_words(dataframe, top_n=5):
    """
    Возвращает топ самых частых слов
    в отрицательных отзывах.
    """
    negative_texts = dataframe[
        dataframe["sentiment"] == 0
    ]["text"].astype(str)

    if negative_texts.empty:
        return []

    count_vectorizer = CountVectorizer()
    word_matrix = count_vectorizer.fit_transform(negative_texts)

    word_counts = np.asarray(
        word_matrix.sum(axis=0)
    ).flatten()

    words = count_vectorizer.get_feature_names_out()

    result = sorted(
        zip(words, word_counts),
        key=lambda item: item[1],
        reverse=True
    )

    return result[:top_n]


negative_words = top_negative_words(df, 5)

print("\n5 САМЫХ ЧАСТЫХ СЛОВ В НЕГАТИВНЫХ ОТЗЫВАХ:")

for word, count in negative_words:
    print(f"{word}: {count}")


# ============================================================
# 3. ЗАЩИТА ПЕРСОНАЛЬНЫХ ДАННЫХ
# ============================================================

print("\nЗАЩИТА ПЕРСОНАЛЬНЫХ ДАННЫХ")


def mask_phone(phone):
    if len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]
    return phone


def mask_name(name):
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)

    return name[:2] + "*" * (len(name) - 4) + name[-2:]


print(
    f"Исходный телефон: {df['phone'].iloc[0]} "
    f"→ {mask_phone(df['phone'].iloc[0])}"
)
print(
    f"Исходное имя: {df['user_name'].iloc[0]} "
    f"→ {mask_name(df['user_name'].iloc[0])}"
)

df_protected = df.copy()
df_protected["user_name"] = df_protected["user_name"].apply(mask_name)
df_protected["phone"] = df_protected["phone"].apply(mask_phone)

print(f"Создана защищённая версия ({len(df_protected)} записей)")


# ============================================================
# 4. ВРЕМЕННЫЕ РЯДЫ
# ============================================================

print("\nВРЕМЕННЫЕ РЯДЫ")

daily_counts = (
    df.groupby(df["date"].dt.date)
    .size()
    .reset_index(name="reviews_count")
)

daily_counts["date"] = pd.to_datetime(daily_counts["date"])
daily_counts = daily_counts.sort_values("date")

daily_counts["day_of_week"] = daily_counts["date"].dt.dayofweek

for lag in [1, 2, 3]:
    daily_counts[f"lag_{lag}"] = daily_counts["reviews_count"].shift(lag)

daily_counts = daily_counts.dropna()

predictions = None
y_test_ts = None

if len(daily_counts) >= 10:
    feature_cols = ["day_of_week", "lag_1", "lag_2", "lag_3"]

    X_train_ts = daily_counts[feature_cols].values[:-3]
    y_train_ts = daily_counts["reviews_count"].values[:-3]

    X_test_ts = daily_counts[feature_cols].values[-3:]
    y_test_ts = daily_counts["reviews_count"].values[-3:]

    ts_model = LinearRegression()
    ts_model.fit(X_train_ts, y_train_ts)

    predictions = ts_model.predict(X_test_ts)
    predictions = np.maximum(0, predictions)

    print("\nПРОГНОЗ НА 3 ДНЯ:")

    for i, (actual, pred) in enumerate(
        zip(y_test_ts, predictions)
    ):
        print(
            f"День {i + 1}: "
            f"Факт = {int(actual):2d} | "
            f"Прогноз = {int(pred):2d}"
        )
else:
    print("Недостаточно данных")


# ============================================================
# 5. РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА
# ============================================================

print("\nРЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")

product_texts = (
    df.groupby("product")
    .agg({"text": lambda x: " ".join(x)})
    .reset_index()
)

rec_vectorizer = TfidfVectorizer(max_features=50)
vectors = rec_vectorizer.fit_transform(product_texts["text"])

similarity_matrix = cosine_similarity(vectors)


def recommend(product_name, top_n=3):
    idx = product_texts[
        product_texts["product"] == product_name
    ].index[0]

    scores = list(
        enumerate(similarity_matrix[idx])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = [
        score
        for score in scores
        if score[0] != idx
    ][:top_n]

    return [
        product_texts.iloc[i]["product"]
        for i, _ in scores
    ]


for product_name in products[:2]:
    print(f"\nРекомендации для '{product_name}':")

    for rec in recommend(product_name):
        print(f"→ {rec}")


# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ И ИТОГОВЫЙ ОТЧЁТ
# ============================================================

print("\nВИЗУАЛИЗАЦИЯ")

# 3 строки × 3 столбца:
# 6 графиков из лекции + 2 новых графика + текстовая статистика
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle(
    "Итоговый анализ отзывов интернет-магазина",
    fontsize=16
)

# 1. Распределение оценок
axes[0, 0].hist(
    df["rating"],
    bins=5,
    edgecolor="black",
    alpha=0.7
)
axes[0, 0].set_title("Распределение оценок")
axes[0, 0].set_xlabel("Оценка")
axes[0, 0].set_ylabel("Количество")

# 2. Динамика отзывов
if len(daily_counts) > 0:
    axes[0, 1].plot(
        daily_counts["date"],
        daily_counts["reviews_count"],
        marker="o",
        linewidth=2
    )

axes[0, 1].set_title("Динамика отзывов")
axes[0, 1].tick_params(
    axis="x",
    rotation=45
)

# 3. Прогноз
if predictions is not None and y_test_ts is not None:
    x_pos = np.arange(3)

    axes[0, 2].bar(
        x_pos - 0.2,
        y_test_ts,
        width=0.4,
        alpha=0.7,
        label="Факт"
    )

    axes[0, 2].bar(
        x_pos + 0.2,
        predictions,
        width=0.4,
        alpha=0.7,
        label="Прогноз"
    )

    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(
        ["День 1", "День 2", "День 3"]
    )
    axes[0, 2].legend()

axes[0, 2].set_title("Прогноз количества отзывов")

# 4. Матрица ошибок
axes[1, 0].imshow(cm)

for i in range(2):
    for j in range(2):
        axes[1, 0].text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

axes[1, 0].set_title("Матрица ошибок")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(
    ["Негатив", "Позитив"]
)
axes[1, 0].set_yticklabels(
    ["Негатив", "Позитив"]
)
axes[1, 0].set_xlabel("Предсказано")
axes[1, 0].set_ylabel("Фактически")

# 5. Популярность товаров
product_counts = df["product"].value_counts()

axes[1, 1].bar(
    product_counts.index,
    product_counts.values,
    alpha=0.7
)
axes[1, 1].set_title("Популярность товаров")
axes[1, 1].tick_params(
    axis="x",
    rotation=45
)

# 6. Соотношение позитив / негатив
sentiment_counts = (
    df_binary["sentiment"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

axes[1, 2].pie(
    sentiment_counts.values,
    labels=["Негатив", "Позитив"],
    autopct="%1.1f%%"
)
axes[1, 2].set_title("Соотношение тональности")

# 7. Топ товаров по доле позитива
positive_chart = top_products.sort_values(
    "positive_percent",
    ascending=True
)

axes[2, 0].barh(
    positive_chart.index,
    positive_chart["positive_percent"]
)
axes[2, 0].set_title(
    "Товары по доле позитивных отзывов"
)
axes[2, 0].set_xlabel("Позитивных отзывов, %")
axes[2, 0].set_xlim(0, 100)

# 8. Топ-5 слов в негативных отзывах
if negative_words:
    neg_words = [item[0] for item in negative_words][::-1]
    neg_counts = [item[1] for item in negative_words][::-1]

    axes[2, 1].barh(
        neg_words,
        neg_counts
    )

axes[2, 1].set_title(
    "5 частых слов в негативных отзывах"
)
axes[2, 1].set_xlabel("Количество")

# 9. Текстовая статистика
axes[2, 2].axis("off")

stats_text = (
    "ОСНОВНАЯ СТАТИСТИКА\n\n"
    f"Всего отзывов: {len(df)}\n"
    f"Размеченных отзывов: {len(df_binary)}\n"
    f"Позитивных: {int(sum(y))}\n"
    f"Негативных: {int(len(y) - sum(y))}\n\n"
    f"Accuracy: {accuracy:.2%}\n"
    f"Precision: {precision:.2%}\n"
    f"Recall: {recall:.2%}\n"
    f"F1-score: {f1:.2%}\n\n"
    f"Лучший товар по доле позитива:\n"
    f"{top_products.index[0]} — "
    f"{top_products.iloc[0]['positive_percent']:.2f}%"
)

axes[2, 2].text(
    0.02,
    0.98,
    stats_text,
    va="top",
    fontsize=11
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96],
    h_pad=3.0,
    w_pad=2.0
)

plt.savefig(
    "student_report.png",
    dpi=150,
    bbox_inches="tight"
)

print("\nИтоговый файл сохранён как student_report.png")

plt.show()
