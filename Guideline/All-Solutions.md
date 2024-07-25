# 📝 Solution Guide for Machine Learning Tasks

## 1. Image Classification using Python and TensorFlow
### Step-by-Step Guide:
1. **Install TensorFlow**: `pip install tensorflow`
2. **Import Libraries**:
    ```python
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    import matplotlib.pyplot as plt
    ```
3. **Load and Preprocess Data**:
    ```python
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    ```
4. **Build the Model**:
    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    ```
5. **Compile and Train the Model**:
    ```python
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    ```
6. **Evaluate the Model**:
    ```python
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_acc}')
    ```
### GitHub Repositories:
- [TensorFlow Examples](https://github.com/tensorflow/examples)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 2. Predicting Student Grades with Linear Regression
### Step-by-Step Guide:
1. **Install Required Libraries**: `pip install pandas scikit-learn`
2. **Import Libraries**:
    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    ```
3. **Load Data**:
    ```python
    data = pd.read_csv('student_grades.csv')
    ```
4. **Preprocess Data**:
    ```python
    X = data[['study_hours', 'sleep_hours']]
    y = data['grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
5. **Train the Model**:
    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```
6. **Make Predictions and Evaluate**:
    ```python
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    ```
### GitHub Repositories:
- [Student Academic Performance Prediction](https://www.javatpoint.com/student-academic-performance-prediction-using-python)

## 3. Text Classification using Naive Bayes
### Step-by-Step Guide:
1. **Install Required Libraries**: `pip install pandas scikit-learn`
2. **Import Libraries**:
    ```python
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    ```
3. **Load Data**:
    ```python
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    ```
4. **Preprocess Data**:
    ```python
    X = data['text']
    y = data['label']
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
5. **Train the Model**:
    ```python
    model = MultinomialNB()
    model.fit(X_train, y_train)
    ```
6. **Make Predictions and Evaluate**:
    ```python
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    ```
### GitHub Repositories:
- [Naive Bayes Classifier Tutorial](https://www.datacamp.com/tutorial/naive-bayes-scikit-learn)
- [Naive Bayes Classifier in Python](https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python)

## 4. Building a Chatbot with Python and NLTK
### Step-by-Step Guide:
1. **Install Required Libraries**: `pip install nltk`
2. **Import Libraries**:
    ```python
    import nltk
    from nltk.chat.util import Chat, reflections
    ```
3. **Define Chatbot Responses**:
    ```python
    pairs = [
        [
            r"my name is (.*)",
            ["Hello %1, How are you today ?",]
        ],
        [
            r"hi|hey|hello",
            ["Hello", "Hey there"]
        ],
        # Add more pairs as needed
    ]
    ```
4. **Initialize and Start Chatbot**:
    ```python
    chat = Chat(pairs, reflections)
    chat.converse()
    ```
### GitHub Repositories:
- [NLTK Documentation](https://www.nltk.org/)

## 5. Sentiment Analysis using Python and Scikit-learn
### Step-by-Step Guide:
1. **Install Required Libraries**: `pip install pandas scikit-learn`
2. **Import Libraries**:
    ```python
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    ```
3. **Load Data**:
    ```python
    data = pd.read_csv('sentiment.csv')
    ```
4. **Preprocess Data**:
    ```python
    X = data['text']
    y = data['label']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
5. **Train the Model**:
    ```python
    model = LogisticRegression()
    model.fit(X_train, y_train)
    ```
6. **Make Predictions and Evaluate**:
    ```python
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    ```
### GitHub Repositories:
- [Sentiment Analysis with Naive Bayes Classifier](https://medium.com/@tpreethi/undesrtand-naive-bayes-algorithm-in-simple-explanation-with-python-code-part-2-a2b91cbbf637)

## 6. Predicting House Prices with Regression
### Step-by-Step Guide:
1. **Install Required Libraries**: `pip install pandas scikit-learn`
2. **Import Libraries**:
    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    ```
3. **Load Data**:
    ```python
    data = pd.read_csv('house_prices.csv')
    ```
4. **Preprocess Data**:
    ```python
    X = data[['square_feet', 'num_rooms', 'age']]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
5. **Train the Model**:
    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```
6. **Make Predictions and Evaluate**:
    ```python
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    ```
### GitHub Repositories:
- [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


## 7. Clustering Customers using K-Means
### Step-by-Step Guide:
1. **Install Required Libraries**: 
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
2. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```
3. **Load Dataset**:
    ```python
    df = pd.read_csv('customer_data.csv')  # replace with your dataset path
    ```
4. **Preprocess Data**:
    ```python
    X = df[['feature1', 'feature2']]  # replace with relevant features
    ```
5. **Determine Optimal Clusters**:
    ```python
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    plt.plot(range(1, 11), inertia)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()
    ```
6. **Train K-Means Model**:
    ```python
    kmeans = KMeans(n_clusters=3, random_state=42)  # choose optimal number of clusters
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_
    ```
7. **Visualize Clusters**:
    ```python
    sns.scatterplot(x='feature1', y='feature2', hue='Cluster', data=df, palette='viridis')
    plt.show()
    ```

### GitHub Repositories:
- [K-Means Clustering](https://www.ris-ai.com/k-mean-clustering-algorithm)

## 8. Building a Recommendation System with Python
### Step-by-Step Guide:
1. **Install Required Libraries**: 
    ```bash
    pip install pandas numpy scikit-learn
    ```
2. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    ```
3. **Load Dataset**:
    ```python
    df = pd.read_csv('user_item_data.csv')  # replace with your dataset path
    ```
4. **Create User-Item Matrix**:
    ```python
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    ```
5. **Calculate Similarity**:
    ```python
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    ```
6. **Make Recommendations**:
    ```python
    def recommend(user_id, num_recommendations):
        similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations+1]
        recommended_items = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False).index
        return recommended_items

    recommendations = recommend(user_id=1, num_recommendations=5)  # example
    print(recommendations)
    ```

### GitHub Repositories:
- [Recommendation System](https://www.geeksforgeeks.org/recommendation-system-in-python/)

## 9. Time Series Forecasting with ARIMA
### Step-by-Step Guide:
1. **Install Required Libraries**: 
    ```bash
    pip install pandas numpy statsmodels matplotlib
    ```
2. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    ```
3. **Load Dataset**:
    ```python
    df = pd.read_csv('time_series_data.csv', parse_dates=True, index_col='Date')  # replace with your dataset path
    ```
4. **Plot Time Series**:
    ```python
    df.plot()
    plt.show()
    ```
5. **Fit ARIMA Model**:
    ```python
    model = sm.tsa.ARIMA(df['value'], order=(5, 1, 0))  # (p,d,q) order
    results = model.fit()
    print(results.summary())
    ```
6. **Make Predictions**:
    ```python
    df['forecast'] = results.predict(start=len(df), end=len(df)+12, typ='levels')  # predict next 12 months
    df[['value', 'forecast']].plot()
    plt.show()
    ```

### GitHub Repositories:
- [ARIMA Time Series Forecasting](https://github.com/aditya1295/Time-Series-Forecasting-ARIMA)

## 10. Object Detection using Python and OpenCV
### Step-by-Step Guide:
1. **Install Required Libraries**: 
    ```bash
    pip install opencv-python
    ```
2. **Import Libraries**:
    ```python
    import cv2
    ```
3. **Load Pre-trained Model**:
    ```python
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # replace with your model paths
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ```
4. **Load Image**:
    ```python
    img = cv2.imread('image.jpg')  # replace with your image path
    height, width, channels = img.shape
    ```
5. **Detect Objects**:
    ```python
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    ```
6. **Display Results**:
    ```python
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### GitHub Repositories:
- [Object Detection with OpenCV](https://github.com/opencv/opencv)

## 🌐 Sources
1. [geeksforgeeks.org - House Price Prediction using Machine Learning in Python](https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/)
2. [javatpoint.com - Predicting Housing Prices using Python](https://www.javatpoint.com/predicting-housing-prices-using-python)
3. [geeksforgeeks.org - Recommendation System in Python](https://www.geeksforgeeks.org/recommendation-system-in-python/)
4. [calistus-igwilo/House-sale
    

