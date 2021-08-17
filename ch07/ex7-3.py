regressor = learn.LinearRegressor(feature_columns=feature_columns,
                                  optimizer=optimizer)
regressor.fit(X, Y, steps=200, batch_size=506)
