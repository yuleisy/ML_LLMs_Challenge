import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

class DelayModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def load_data(self, file_path):
        # Carga los datos desde un archivo CSV
        data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
        return data

    def preprocess_data(self, data):
        # Limpia columnas de fecha
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

        # Elimina filas con NaT en las fechas
        data.dropna(subset=['Fecha-I', 'Fecha-O'], inplace=True)

        # Crea las nuevas columnas requeridas
        data['high_season'] = ((data['Fecha-I'].dt.month == 12) & (data['Fecha-I'].dt.day >= 15)) | \
                              ((data['Fecha-I'].dt.month == 3) & (data['Fecha-I'].dt.day <= 3)) | \
                              ((data['Fecha-I'].dt.month == 7) & (data['Fecha-I'].dt.day >= 15)) | \
                              ((data['Fecha-I'].dt.month == 7) & (data['Fecha-I'].dt.day <= 31)) | \
                              ((data['Fecha-I'].dt.month == 9) & (data['Fecha-I'].dt.day >= 11)) | \
                              ((data['Fecha-I'].dt.month == 9) & (data['Fecha-I'].dt.day <= 30))
        data['high_season'] = data['high_season'].astype(int)

        # Calcula min_diff
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60

        # Asigna periodos del día
        bins = [-1, 5, 12, 19, 24]
        labels = ['night', 'morning', 'afternoon', 'night_end']
        data['period_day'] = pd.cut(data['Fecha-I'].dt.hour, bins=bins, labels=labels, right=True, include_lowest=True)

        data['delay'] = (data['min_diff'] > 15).astype(int)

        # Maneja columnas faltantes
        required_columns = ['high_season', 'min_diff', 'DIA', 'MES', 'AÑO', 'period_day', 'delay']
        for col in required_columns:
            if col not in data.columns:
                data[col] = 0  # Agrega columna faltante con valores por defecto (0)

        return data

    def train(self, file_path):
        # Carga y procesa los datos
        data = self.load_data(file_path)
        data = self.preprocess_data(data)

        # Define las características y el objetivo
        features = data[['high_season', 'min_diff', 'DIA', 'MES', 'AÑO']]
        target = data['delay']

        # Divide el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Entrena el modelo
        self.model.fit(X_train, y_train)

        # Evalua el modelo
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Matriz de confusión y reporte de clasificación
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        print(f'Model accuracy: {accuracy:.2f}')
        print('Confusion Matrix:')
        print(conf_matrix)
        print('Classification Report:')
        print(class_report)

    def save_model(self, file_path):
        # Guarda el modelo y las columnas
        with open(file_path, 'wb') as f:
            # Guarda el modelo y las columnas relevantes
            pickle.dump({'model': self.model, 'columns': ['high_season', 'min_diff', 'DIA', 'MES', 'AÑO', 'period_day', 'delay']}, f)

# Ejecución del modelo
if __name__ == "__main__":
    delay_model = DelayModel()
    file_path = 'C:/Users/yuleisy.zamora/Documents/ML_LLMs_Challenge/data/data.csv' 
    delay_model.train(file_path)  
    delay_model.save_model('C:/Users/yuleisy.zamora/Documents/ML_LLMs_Challenge/model_entrenado.pkl')  # Guarda el modelo
