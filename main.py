import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
                             QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit,
                             QHBoxLayout, QFormLayout, QGroupBox, QTableWidget,
                             QTableWidgetItem, QTabWidget, QProgressBar, QTextEdit,
                             QComboBox, QCheckBox, QTableView, QHeaderView)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTime, Qt, QAbstractTableModel
import matplotlib.pyplot as plt
import logging
import os

# Настройка логирования
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'app.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class TrainThread(QThread):
    progress_updated = pyqtSignal(int)
    training_finished = pyqtSignal(float, float, np.ndarray, np.ndarray, float)

    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, batch_size):
        QThread.__init__(self)
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.stop_training = False

    def run(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        start_time = QTime.currentTime()
        for epoch in range(self.epochs):
            if self.stop_training:
                break
            self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=self.batch_size, verbose=1, validation_data=(self.X_val, self.y_val), callbacks=[early_stopping])
            self.progress_updated.emit((epoch + 1) * 100 // self.epochs)
        end_time = QTime.currentTime()
        elapsed_time = start_time.msecsTo(end_time) / 1000.0  # Время в секундах
        predictions = self.model.predict(self.X_val)
        mse = mean_squared_error(self.y_val, predictions)
        r2 = r2_score(self.y_val, predictions)
        self.training_finished.emit(mse, r2, self.y_test, self.model.predict(self.X_test), elapsed_time)

    def stop(self):
        self.stop_training = True

class CarBreakdownPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Прогнозирование поломок автомобилей")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Вкладка для работы с файлами
        self.files_tab = QWidget()
        self.tabs.addTab(self.files_tab, "Файлы")
        self.files_layout = QVBoxLayout()
        self.files_layout.setSpacing(5)  # Уменьшаем вертикальные промежутки
        self.files_tab.setLayout(self.files_layout)

        self.generate_data_size_input = QComboBox()
        self.generate_data_size_input.addItems(["1000", "10 000", "100 000", "1 000 000", "10 000 000"])
        self.files_layout.addWidget(self.generate_data_size_input)

        self.load_data_button = QPushButton("Загрузить данные для обучения")
        self.load_data_button.clicked.connect(self.load_data)
        self.files_layout.addWidget(self.load_data_button)

        self.generate_data_button = QPushButton("Сгенерировать данные для обучения")
        self.generate_data_button.clicked.connect(self.generate_data)
        self.files_layout.addWidget(self.generate_data_button)

        self.view_data_button = QPushButton("Просмотреть данные")
        self.view_data_button.clicked.connect(self.view_data)
        self.files_layout.addWidget(self.view_data_button)

        self.clear_data_button = QPushButton("Удалить сгенерированные данные")
        self.clear_data_button.clicked.connect(self.clear_data)
        self.files_layout.addWidget(self.clear_data_button)

        self.open_data_folder_button = QPushButton("Открыть папку с данными для обучения")
        self.open_data_folder_button.clicked.connect(self.open_data_folder)
        self.files_layout.addWidget(self.open_data_folder_button)

        self.open_models_folder_button = QPushButton("Открыть папку с моделями")
        self.open_models_folder_button.clicked.connect(self.open_models_folder)
        self.files_layout.addWidget(self.open_models_folder_button)

        self.open_logs_folder_button = QPushButton("Открыть папку с логами")
        self.open_logs_folder_button.clicked.connect(self.open_logs_folder)
        self.files_layout.addWidget(self.open_logs_folder_button)

        # Вкладка для загрузки данных и обучения модели
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Данные и Обучение")
        self.data_layout = QVBoxLayout()
        self.data_tab.setLayout(self.data_layout)

        self.train_button = QPushButton("Обучить модель")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        self.data_layout.addWidget(self.train_button)

        self.stop_button = QPushButton("Завершить обучение")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.data_layout.addWidget(self.stop_button)

        self.save_model_button = QPushButton("Сохранить модель")
        self.save_model_button.clicked.connect(self.save_model)
        self.save_model_button.setEnabled(False)
        self.data_layout.addWidget(self.save_model_button)

        self.load_model_button = QPushButton("Загрузить модель")
        self.load_model_button.clicked.connect(self.load_model_file)
        self.data_layout.addWidget(self.load_model_button)

        self.progress_label = QLabel("Прогресс обучения модели:")
        self.data_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.data_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.data_layout.addWidget(self.log_text)

        # Вкладка для прогнозирования
        self.predict_tab = QWidget()
        self.tabs.addTab(self.predict_tab, "Прогнозирование")
        self.predict_layout = QVBoxLayout()
        self.predict_tab.setLayout(self.predict_layout)

        self.input_group = QGroupBox("Ввод данных для прогнозирования")
        self.input_layout = QFormLayout()

        self.age_input = QComboBox()
        self.age_input.addItems([str(i) for i in range(1, 21)])
        self.mileage_input = QLineEdit()
        self.breakdown_frequency_input = QLineEdit()
        self.additional_param_input = QCheckBox("Использование премиального топлива")
        self.additional_param_input2 = QCheckBox("Регулярное техническое обслуживание")
        self.additional_param_input3 = QCheckBox("Использование качественных запчастей")
        self.winter_use_input = QComboBox()
        self.winter_use_input.addItems(["Да", "Нет"])
        self.region_input = QComboBox()
        self.region_input.addItems(["Центральный", "Северо-Западный", "Южный", "Северо-Кавказский", "Приволжский", "Уральский", "Сибирский", "Дальневосточный"])
        self.usage_frequency_input = QLineEdit()
        self.engine_hours_input = QLineEdit()

        self.input_layout.addRow("Возраст автомобиля (лет):", self.age_input)
        self.input_layout.addRow("Пробег (км):", self.mileage_input)
        self.input_layout.addRow("Частота поломок в год:", self.breakdown_frequency_input)
        self.input_layout.addRow(self.additional_param_input)
        self.input_layout.addRow(self.additional_param_input2)
        self.input_layout.addRow(self.additional_param_input3)
        self.input_layout.addRow("Использование в зимний период:", self.winter_use_input)
        self.input_layout.addRow("Регион использования:", self.region_input)
        self.input_layout.addRow("Частота использования (раз в неделю):", self.usage_frequency_input)
        self.input_layout.addRow("Моточасы (часы за последний месяц):", self.engine_hours_input)

        self.input_group.setLayout(self.input_layout)
        self.predict_layout.addWidget(self.input_group)

        self.predict_button = QPushButton("Прогнозировать")
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setEnabled(False)
        self.predict_layout.addWidget(self.predict_button)

        self.result_label = QLabel("")
        self.predict_layout.addWidget(self.result_label)

        # Вкладка для истории прогнозов
        self.history_tab = QWidget()
        self.tabs.addTab(self.history_tab, "История Прогнозов")
        self.history_layout = QVBoxLayout()
        self.history_tab.setLayout(self.history_layout)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(10)
        self.history_table.setHorizontalHeaderLabels(["Возраст", "Пробег", "Частота поломок в год", "Доп. параметр 1", "Доп. параметр 2", "Доп. параметр 3", "Зимний период", "Регион", "Частота использования", "Прогноз"])
        self.history_layout.addWidget(self.history_table)

        self.plot_label = QLabel()
        self.data_layout.addWidget(self.plot_label)

        self.history = []
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.train_thread = None

        # Загрузка данных при старте
        self.load_data_on_startup()

        # Обработчик клика по графику
        self.plot_label.mousePressEvent = self.show_formulas

    def closeEvent(self, event):
        self.save_data_on_exit()
        event.accept()

    def load_data_on_startup(self):
        data_path = "data/generated_data.csv"
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            self.train_button.setEnabled(True)
            logging.info("Data loaded from file on startup.")

    def save_data_on_exit(self):
        if hasattr(self, 'df'):
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            self.df.to_csv(os.path.join(data_dir, "generated_data.csv"), index=False)
            logging.info("Data saved on exit.")

    def load_data(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить данные", "", "CSV Files (*.csv)")
            if file_path:
                self.df = pd.read_csv(file_path)
                self.train_button.setEnabled(True)
                QMessageBox.information(self, "Успех", "Данные успешно загружены!")
                logging.info("Data loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке данных: {e}")
            logging.error(f"Error loading data: {e}")

    def generate_data(self):
        try:
            data_size = self.generate_data_size_input.currentText()
            if data_size == "1000":
                num_samples = 1000
            elif data_size == "10 000":
                num_samples = 10000
            elif data_size == "100 000":
                num_samples = 100000
            elif data_size == "1 000 000":
                num_samples = 1000000
            elif data_size == "10 000 000":
                num_samples = 10000000
            else:
                QMessageBox.critical(self, "Ошибка", "Неверный размер данных!")
                return

            np.random.seed(42)
            ages = np.random.randint(1, 21, num_samples)
            mileages = np.random.randint(10000, 200000, num_samples)
            breakdown_frequencies = np.random.randint(1, 10, num_samples)
            additional_param1 = np.random.choice([0, 1], num_samples)
            additional_param2 = np.random.choice([0, 1], num_samples)
            additional_param3 = np.random.choice([0, 1], num_samples)
            winter_use = np.random.choice(["Да", "Нет"], num_samples)
            region = np.random.choice(["Центральный", "Северо-Западный", "Южный", "Северо-Кавказский", "Приволжский", "Уральский", "Сибирский", "Дальневосточный"], num_samples)
            usage_frequency = np.random.randint(1, 8, num_samples)
            engine_hours = np.random.randint(100, 2000, num_samples)

            # Учет того, что у новых машин шанс поломок меньше
            breakdowns = np.random.randint(0, 20, num_samples)
            breakdowns = np.where(ages < 5, breakdowns * 0.5, breakdowns)

            self.df = pd.DataFrame({
                'age': ages,
                'mileage': mileages,
                'breakdown_frequency': breakdown_frequencies,
                'additional_param1': additional_param1,
                'additional_param2': additional_param2,
                'additional_param3': additional_param3,
                'winter_use': winter_use,
                'region': region,
                'usage_frequency': usage_frequency,
                'engine_hours': engine_hours,
                'breakdowns': breakdowns
            })

            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            self.df.to_csv(os.path.join(data_dir, "generated_data.csv"), index=False)
            self.train_button.setEnabled(True)
            QMessageBox.information(self, "Успех", "Данные успешно сгенерированы!")
            logging.info("Data generated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при генерации данных: {e}")
            logging.error(f"Error generating data: {e}")

    def view_data(self):
        try:
            if not hasattr(self, 'df'):
                QMessageBox.critical(self, "Ошибка", "Сначала загрузите или сгенерируйте данные!")
                return

            self.data_view = QTableView()
            model = PandasModel(self.df)
            self.data_view.setModel(model)
            self.data_view.setWindowTitle("Просмотр данных")
            self.data_view.resize(800, 600)
            self.data_view.show()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при просмотре данных: {e}")
            logging.error(f"Error viewing data: {e}")

    def clear_data(self):
        try:
            if hasattr(self, 'df'):
                del self.df
                self.train_button.setEnabled(False)
                QMessageBox.information(self, "Успех", "Данные успешно удалены!")
                logging.info("Data cleared successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при удалении данных: {e}")
            logging.error(f"Error clearing data: {e}")

    def open_data_folder(self):
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            os.startfile(data_dir)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при открытии папки с данными: {e}")
            logging.error(f"Error opening data folder: {e}")

    def open_models_folder(self):
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            os.startfile(models_dir)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при открытии папки с моделями: {e}")
            logging.error(f"Error opening models folder: {e}")

    def open_logs_folder(self):
        try:
            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            os.startfile(logs_dir)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при открытии папки с логами: {e}")
            logging.error(f"Error opening logs folder: {e}")

    def start_training(self):
        try:
            if not hasattr(self, 'df'):
                QMessageBox.critical(self, "Ошибка", "Сначала загрузите или сгенерируйте данные!")
                return

            self.preprocess_data()

            X = self.df.drop('breakdowns', axis=1).values
            y = self.df['breakdowns'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

            self.model = Sequential()
            self.model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dense(1))

            self.model.compile(optimizer='adam', loss='mean_squared_error')

            self.train_thread = TrainThread(self.model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=200, batch_size=8)
            self.train_thread.progress_updated.connect(self.update_progress)
            self.train_thread.training_finished.connect(self.training_finished)
            self.train_thread.start()

            self.stop_button.setEnabled(True)
            logging.info("Training started.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при начале обучения: {e}")
            logging.error(f"Error starting training: {e}")

    def stop_training(self):
        try:
            if self.train_thread is not None:
                self.train_thread.stop()
                self.stop_button.setEnabled(False)
                logging.info("Training stopped.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при остановке обучения: {e}")
            logging.error(f"Error stopping training: {e}")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def training_finished(self, mse, r2, y_test, predictions, elapsed_time):
        try:
            self.progress_bar.setValue(100)
            self.stop_button.setEnabled(False)
            self.save_model_button.setEnabled(True)
            self.predict_button.setEnabled(True)

            # Вывод результатов погрешности и времени обучения
            self.log_text.append(f"MSE: {mse:.4f}")
            self.log_text.append(f"R2 Score: {r2:.4f}")
            self.log_text.append(f"Время обучения: {elapsed_time:.2f} секунд")

            # Визуализация результатов
            plt.scatter(y_test, predictions)
            plt.xlabel("Реальные значения")
            plt.ylabel("Предсказанные значения")
            plt.title("Реальные vs Предсказанные значения")
            plt.savefig("predictions.png")
            self.plot_label.setPixmap(QPixmap("predictions.png"))

            logging.info("Training finished.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при завершении обучения: {e}")
            logging.error(f"Error finishing training: {e}")

    def save_model(self):
        try:
            if self.model is None:
                QMessageBox.critical(self, "Ошибка", "Сначала обучите модель!")
                return

            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", models_dir, "H5 Files (*.h5)")
            if file_path:
                self.model.save(file_path)
                QMessageBox.information(self, "Успех", "Модель успешно сохранена!")
                logging.info("Model saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении модели: {e}")
            logging.error(f"Error saving model: {e}")

    def load_model_file(self):
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить модель", models_dir, "H5 Files (*.h5)")
            if file_path:
                self.model = load_model(file_path)
                self.predict_button.setEnabled(True)
                QMessageBox.information(self, "Успех", "Модель успешно загружена!")
                logging.info("Model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке модели: {e}")
            logging.error(f"Error loading model: {e}")

    def predict(self):
        try:
            if self.model is None:
                QMessageBox.critical(self, "Ошибка", "Сначала обучите или загрузите модель!")
                return

            age = int(self.age_input.currentText())
            mileage = float(self.mileage_input.text())
            breakdown_frequency = float(self.breakdown_frequency_input.text())
            additional_param1 = int(self.additional_param_input.isChecked())
            additional_param2 = int(self.additional_param_input2.isChecked())
            additional_param3 = int(self.additional_param_input3.isChecked())
            winter_use = self.winter_use_input.currentText()
            region = self.region_input.currentText()
            usage_frequency = float(self.usage_frequency_input.text())
            engine_hours = float(self.engine_hours_input.text())

            winter_use_encoded = self.label_encoders['winter_use'].transform([winter_use])[0]
            region_encoded = self.label_encoders['region'].transform([region])[0]

            input_data = np.array([[age, mileage, breakdown_frequency, additional_param1, additional_param2, additional_param3, winter_use_encoded, region_encoded, usage_frequency, engine_hours]])
            input_data = self.scaler.transform(input_data)
            prediction = self.model.predict(input_data)
            self.result_label.setText(f'Предсказанное количество поломок: {prediction[0][0]:.2f} (в среднем за месяц)')

            # Сохранение истории
            self.history.append([age, mileage, breakdown_frequency, additional_param1, additional_param2, additional_param3, winter_use, region, usage_frequency, prediction[0][0]])
            self.update_history_table()
            logging.info(f"Prediction made: {prediction[0][0]:.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при прогнозировании: {e}")
            logging.error(f"Error making prediction: {e}")

    def update_history_table(self):
        try:
            self.history_table.setRowCount(len(self.history))
            for i, row in enumerate(self.history):
                for j, value in enumerate(row):
                    self.history_table.setItem(i, j, QTableWidgetItem(str(value)))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обновлении истории: {e}")
            logging.error(f"Error updating history: {e}")

    def preprocess_data(self):
        try:
            self.label_encoders['winter_use'] = LabelEncoder()
            self.df['winter_use'] = self.label_encoders['winter_use'].fit_transform(self.df['winter_use'])

            self.label_encoders['region'] = LabelEncoder()
            self.df['region'] = self.label_encoders['region'].fit_transform(self.df['region'])

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при предобработке данных: {e}")
            logging.error(f"Error preprocessing data: {e}")

    def show_formulas(self, event):
        # Отображение формул
        formula_real = "Реальные данные: y = f(x)"
        formula_predicted = "Прогноз: y_pred = f(x)"

        QMessageBox.information(self, "Формулы", f"{formula_real}\n{formula_predicted}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CarBreakdownPredictionApp()
    main_window.show()
    sys.exit(app.exec_())
