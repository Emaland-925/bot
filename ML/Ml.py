class CityWeatherData:
    def __init__(self, city):
        self.city = city

    def analyze_weather(self, ALLSKY, CLRSKY, temperature, pressure, moisture):
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import train_test_split

        file_path = f'ML/{self.city.capitalize()} 2020-2022.csv'
        data = pd.read_csv(file_path)

        X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values
        y = data.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

        reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], fit_intercept=False, cv=10).fit(X_train, y_train)

        # Prepare new data for prediction
        new_data = np.array([[float(ALLSKY), float(CLRSKY), float(pressure), float(temperature), float(moisture)]])

        # Make predictions
        predictions_linear = reg.predict(new_data)

        return predictions_linear








