import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .config import TEST_SPLIT, RANDOM_STATE


class StockPredictor:
    """
    Train and manage linear regression models for stock price prediction
    """

    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def train(self, features, target):
        """
        Train the linear regression model with evaluation

        Args:
            features: pandas DataFrame or numpy array with features
            target: pandas Series or numpy array with target prices

        Returns:
            dict: Training metrics (MAE, RMSE, R²)
        """
        print(f"\n🧠 Training {self.ticker} model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=TEST_SPLIT, random_state=RANDOM_STATE
        )

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Scale features (important for linear regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

        # Store feature names
        self.feature_names = list(features.columns)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        directional_accuracy = self._calculate_directional_accuracy(y_test.values, y_pred)

        self.metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()

        print("📊 Model Performance:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
        print(f"  Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        # Save model if it meets minimum criteria
        if r2 > 0.5:  # Only save reasonably good models
            self.save_model()
        else:
            print(f"⚠️ {self.ticker} model performance too low (R² = {r2:.3f}), not saving")

        return self.metrics

    def _calculate_directional_accuracy(self, actual, predicted):
        """
        Calculate percentage of times the model predicted the correct direction
        """
        actual_direction = np.diff(actual) > 0  # True if price went up
        pred_direction = np.diff(predicted) > 0  # True if prediction was up

        correct_direction = np.sum(actual_direction == pred_direction)
        total_predictions = len(actual_direction)

        return (correct_direction / total_predictions) * 100 if total_predictions > 0 else 0

    def predict(self, features):
        """
        Make prediction on new data

        Args:
            features: pandas DataFrame with feature columns matching training

        Returns:
            float: Predicted stock price
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load_model().")

        # Scale features using the same scaler
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)

        return prediction[0] if len(prediction) == 1 else prediction

    def predict_with_confidence(self, features, confidence=0.95):
        """
        Make prediction with confidence interval

        Args:
            features: feature DataFrame
            confidence: confidence level (default 95%)

        Returns:
            dict: Prediction with upper/lower bounds
        """
        prediction = self.predict(features)

        # Simple confidence interval based on historical MAPE
        if 'mape' in self.metrics:
            error_margin = (self.metrics['mape'] / 100) * prediction
            margin = error_margin / 1.96  # Assuming normal distribution

            return {
                'prediction': prediction,
                'lower_bound': prediction - margin,
                'upper_bound': prediction + margin,
                'confidence_interval': margin * 2,
                'confidence_level': confidence
            }
        else:
            return {'prediction': prediction}

    def get_feature_importance(self):
        """
        Get feature importance from linear regression coefficients

        Returns:
            dict: Feature names mapped to their coefficients
        """
        if self.model is None or self.feature_names is None:
            return {}

        coefficients = self.model.coef_
        importance = dict(zip(self.feature_names, coefficients))

        # Sort by absolute value for most important features
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

        return dict(sorted_features)

    def save_model(self):
        """
        Save trained model and scaler to disk
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'ticker': self.ticker
        }

        filename = f"{self.models_dir}/{self.ticker}_model.pkl"
        joblib.dump(model_data, filename)

        print(f"💾 Model saved to {filename}")

    @classmethod
    def load_model(cls, ticker):
        """
        Load a trained model from disk
        """
        models_dir = "models"
        filename = f"{models_dir}/{ticker}_model.pkl"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        model_data = joblib.load(filename)

        # Create instance and restore state
        instance = cls(ticker)
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data.get('metrics', {})

        print(f"📂 Loaded {ticker} model from disk (R²: {instance.metrics.get('r2', 0):.3f})")

        return instance

    def plot_feature_importance(self, top_n=10):
        """
        Plot the most important features
        """
        if self.model is None:
            print("No trained model to plot")
            return

        importance = self.get_feature_importance()
        top_features = dict(list(importance.items())[:top_n])

        plt.figure(figsize=(10, 6))
        features = list(top_features.keys())
        coeffs = [abs(coeff) for coeff in top_features.values()]

        plt.barh(features[::-1], coeffs[::-1])
        plt.title(f'{self.ticker} - Top {top_n} Important Features')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()

        return plt.gcf()
