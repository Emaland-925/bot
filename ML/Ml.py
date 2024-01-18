#!/usr/bin/env python
# coding: utf-8

# In[12]:


#NEW
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





'''
    def jeddah(self):
        import pandas as pd
        
        file_path = 'jeddah 2020-2022.csv'
        data = pd.read_csv(file_path)
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        
        X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values
        y = data.iloc[:, -1].values

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

       
        scaler = StandardScaler()

        # Fit the scaler on the training data and transform both training and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        
        from sklearn.linear_model import RidgeCV

        reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                       fit_intercept=False,cv=10).fit(X_train, y_train)

        y_pred2 = reg2.predict(X_test)
        
        
        #new_data = np.array([[weather_data.ALLSKY, weather_data.CLRSKY, weather_data.pressure, weather_data.temperature, weather_data.moisture]])
        new_data = np.array([[float(self.ALLSKY), float(self.CLRSKY), float(self.pressure), float(self.temperature), float(self.moisture)]])

       
        predictions_linear = reg2.predict(new_data)

 
        print("The Solar Energy is: ", predictions_linear)
        
    def skaka(self):
        import pandas as pd
        
        file_path = 'SKAKA 2020-2022.csv'
        data = pd.read_csv(file_path)
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        
        X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values
        y = data.iloc[:, -1].values

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
        scaler = StandardScaler()

        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        
        from sklearn.linear_model import RidgeCV

        reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                       fit_intercept=False,cv=10).fit(X_train, y_train)

        y_pred2 = reg2.predict(X_test)
        
        
        #new_data = np.array([[weather_data.ALLSKY, weather_data.CLRSKY, weather_data.pressure, weather_data.temperature, weather_data.moisture]])
        new_data = np.array([[float(self.ALLSKY), float(self.CLRSKY), float(self.pressure), float(self.temperature), float(self.moisture)]])

        
        predictions_linear = reg2.predict(new_data)

        
        print("The Solar Energy is: ", predictions_linear)

    def riyadh(self):
        import pandas as pd
        
        file_path = 'Riyadh 2020-2022.csv'
        data = pd.read_csv(file_path)
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        
        X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values
        y = data.iloc[:, -1].values

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        
        scaler = StandardScaler()

        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
       
        from sklearn.linear_model import RidgeCV

        reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                       fit_intercept=False,cv=10).fit(X_train, y_train)

        y_pred2 = reg2.predict(X_test)
        
       
        #new_data = np.array([[weather_data.ALLSKY, weather_data.CLRSKY, weather_data.pressure, weather_data.temperature, weather_data.moisture]])
        new_data = np.array([[float(self.ALLSKY), float(self.CLRSKY), float(self.pressure), float(self.temperature), float(self.moisture)]])

        
        predictions_linear = reg2.predict(new_data)

        
        print("The Solar Energy is: ", predictions_linear)


    def dammam(self):
        import pandas as pd
        
        file_path = 'DAmmam 2020-2022.csv'
        data = pd.read_csv(file_path)
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

       
        X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values
        y = data.iloc[:, -1].values

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        
        scaler = StandardScaler()

        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        
        from sklearn.linear_model import RidgeCV

        reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                       fit_intercept=False,cv=10).fit(X_train, y_train)

        y_pred2 = reg2.predict(X_test)
        
       
        #new_data = np.array([[weather_data.ALLSKY, weather_data.CLRSKY, weather_data.pressure, weather_data.temperature, weather_data.moisture]])
        new_data = np.array([[float(self.ALLSKY), float(self.CLRSKY), float(self.pressure), float(self.temperature), float(self.moisture)]])

        
        predictions_linear = reg2.predict(new_data)

        
        print("The Solar Energy is: ", predictions_linear)



print("Please Enter a city:")
print("1 - Jeddah")
print("2 - Skaka")
print("3 - Riyadh")
print("4 - Dammam")


city_choice = int(input("Enter the number corresponding to the city: "))


weather_data = CityWeatherData()


weather_data.ALLSKY = input("Enter ALLSKY value: ")
weather_data.CLRSKY = input("Enter CLRSKY value: ")
weather_data.temperature = float(input("Enter temperature: "))
weather_data.pressure = float(input("Enter pressure: "))
weather_data.moisture = float(input("Enter moisture: "))


if city_choice == 1:
    weather_data.jeddah()
elif city_choice == 2:
    weather_data.skaka()
elif city_choice == 3:
    weather_data.riyadh()
elif city_choice == 4:
    weather_data.dammam()
else:
    print("Invalid choice. Please choose a number between 1 and 4.")


# In[ ]:
'''


