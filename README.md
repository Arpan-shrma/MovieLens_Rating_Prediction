# 🎬 MovieLens Rating Prediction Project

## 🎯 Project Overview
A machine learning project to predict movie ratings using the MovieLens dataset. This project implements an XGBoost-based recommendation system with extensive feature engineering.

## 📊 Data Description
- Training set: 90,836 ratings
- Test set: 10,000 ratings
- Features include: user data, movie data, genres, and temporal information

## 🛠️ Feature Engineering

### User-Based Features
- Average rating per user
- Rating standard deviation
- Total ratings count
- Genre preferences

### Movie-Based Features
- Average movie rating
- Rating standard deviation
- Popularity metrics
- Genre encodings

### Genre Engineering
- Multi-label binarization
- One-hot encoding
- Genre combination analysis

## 🔧 Data Preprocessing Steps

1. **Data Cleaning**
   - Handled missing values
   - Removed duplicates
   - Standardized formats

2. **Feature Creation**
   ```python
   # Example of feature creation
   movie_stats = train_data_merged.groupby('movieId').agg({
       'rating': ['mean', 'std', 'count']
   })
   ```

3. **Text Processing**
   - Genre parsing
   - Title extraction
   - Category normalization

## 🤖 Model Architecture

### XGBoost Implementation
```python
best_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### Hyperparameter Tuning
- Grid search with cross-validation
- Parameter ranges explored:
  - max_depth: [6, 8]
  - eta: [0.01, 0.1, 0.2]
  - subsample: [0.6, 0.8]
  - colsample_bytree: [0.6, 0.8]

## 📈 Model Performance

### Training Metrics
- Best validation RMSE: 0.7425
- Early stopping at ~150 iterations
- Learning rate: 0.1

### Test Results
- Leaderboard RMSE: 0.81 (per hundredth of a star)
- Effective on various movie genres
- Robust to user rating patterns

## 📊 Visualizations

The project includes several key visualizations:
- Training vs Validation loss curves
- Genre distribution analysis
- Rating distribution patterns
- Feature importance plots

## 🚀 Usage

1. Data Preparation:
```python
# Load and preprocess data
links_df = pd.read_csv("links.csv")
test_df = pd.read_csv("movie_ratings_test.csv")
train_df = pd.read_csv("movie_ratings_train.csv")
movie_name_df = pd.read_csv("movies.csv")
```

2. Feature Engineering:
```python
# Create user stats
user_stats = train_data_merged.groupby('userId').agg({
    'rating': ['mean', 'std', 'count']
})
```

3. Model Training:
```python
model = xgb.train(best_params, dtrain, 
                 num_boost_round=500,
                 early_stopping_rounds=50)
```

## 📝 Dependencies
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## 🎓 Academic Context
Submitted to University of Guelph for DATA*6100.

## 👥 Authors
Contact for questions or contributions.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- MovieLens dataset
- University of Guelph
- XGBoost community
