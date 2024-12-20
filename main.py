import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Read the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def random_forest_model(n_est, rand_stat):
    return RandomForestRegressor(n_estimators=n_est, random_state=rand_stat)

# Function for comparing different models
def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)



if __name__ == '__main__':

    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = train_data[features].copy()
    X_test = test_data[features].copy()

    # Target
    y = train_data.SalePrice

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    models = []
    for i in range(20,210,20):
        models.append(random_forest_model(i,0))

    for i in range(0, len(models)):
        mae = score_model(models[i], X_train, X_valid, y_train, y_valid)
        print("Model %d MAE: %d" % (i + 1, mae))


    # Generate test predictions
    main_model = RandomForestRegressor(random_state=0)

    # Fit the model to the training data
    main_model.fit(X, y)

    # Generate test predictions
    preds_test = main_model.predict(X_test)

    # Save predictions in format used for competition scoring
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})


    print(output)



