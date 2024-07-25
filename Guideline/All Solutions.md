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
1. **Install Required Libraries**: `pip install pandas scikit-learn`
2. **Import Libraries**:
    ```python
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    ```
3. **Load Data**:
    ```python
    data = pd.read_csv('customers.csv')
    ```
4. **Preprocess Data**:
    ```python
    X = data[['age', 'income', 'spending_score']]
    ```
5. **Train the Model**:
    ```python
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    data['cluster'] = kmeans.labels_
    ```
6. **Visualize the
