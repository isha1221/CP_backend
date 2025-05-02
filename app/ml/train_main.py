import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs("app/ml", exist_ok=True)
os.makedirs("app/reports", exist_ok=True)


def timer_decorator(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper


@timer_decorator
def load_and_explore_data(file_path):
    """Load and explore the dataset"""
    print(f"\n{'='*20} LOADING AND EXPLORING DATA {'='*20}")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique job roles: {df['Job Title'].nunique()}")
    print(f"Top 10 most common roles:")
    print(df['Job Title'].value_counts().head(10))

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values per column:")
        print(missing_values[missing_values > 0])

        # Handle missing values
        print("Handling missing values...")
        df['Key Skills'] = df['Key Skills'].fillna('')
        df['Job Title'] = df['Job Title'].fillna('Unknown')

    # Check text length distribution
    df['text_length'] = df['Key Skills'].apply(lambda x: len(str(x).split()))
    print("\nText length statistics:")
    print(df['text_length'].describe())

    # Visualize role distribution
    plt.figure(figsize=(12, 6))
    role_counts = df['Job Title'].value_counts()
    if role_counts.shape[0] > 15:
        sns.barplot(x=role_counts.index[:15], y=role_counts.values[:15])
        plt.title('Distribution of Top 15 Job Roles')
    else:
        sns.barplot(x=role_counts.index, y=role_counts.values)
        plt.title('Distribution of Job Roles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('app/reports/role_distribution.png')
    plt.close()

    # Check for class imbalance
    imbalance_ratio = role_counts.max() / role_counts.min()
    print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}")

    # Save additional insights
    rare_roles = role_counts[role_counts < 5].index.tolist()
    if rare_roles:
        print(
            f"\nWarning: {len(rare_roles)} roles have fewer than 5 examples each.")
        print(
            f"Consider combining rare roles or collecting more data for these categories.")

    return df


@timer_decorator
def preprocess_text(texts):
    """Advanced text preprocessing function"""
    # Download NLTK resources
    for resource in ['stopwords', 'wordnet', 'punkt']:
        try:
            nltk.download(resource, quiet=True)
        except:
            print(
                f"Warning: Could not download NLTK resource {resource}. Some preprocessing features may be limited.")

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Common job-related abbreviations mapping
    abbreviations = {
        'sr': 'senior',
        'jr': 'junior',
        'mgr': 'manager',
        'dev': 'developer',
        'engg': 'engineering',
        'eng': 'engineer',
        'prog': 'programmer',
        'admin': 'administrator',
        'coord': 'coordinator',
        'dir': 'director',
        'exec': 'executive',
        'hr': 'human resources',
        'it': 'information technology',
        'qa': 'quality assurance',
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'ds': 'data science',
        'pm': 'project manager',
        'ui': 'user interface',
        'ux': 'user experience'
    }

    # Domain-specific stopwords to remove
    try:
        domain_stopwords = set(stopwords.words('english'))
        # Keep important job-related terms
        keep_words = {'senior', 'junior', 'lead', 'manager', 'director', 'head', 'chief',
                      'specialist', 'executive', 'professional', 'experience', 'expert'}
        domain_stopwords = domain_stopwords - keep_words
    except:
        domain_stopwords = set()
        print("Warning: Could not load stopwords. Proceeding without stopword removal.")

    print("Preprocessing text data...")
    processed_texts = []

    for text in texts:
        if not isinstance(text, str):
            processed_texts.append("")
            continue

        # Convert to lowercase
        text = text.lower()

        # Expand abbreviations
        for abbr, full in abbreviations.items():
            pattern = r'\b' + abbr + r'\b'
            text = re.sub(pattern, full, text)

        # Remove special characters but keep hyphens between words
        text = re.sub(r'[^\w\s\-]', ' ', text)

        # Replace hyphens with spaces
        text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', text)

        # Remove numbers
        text = re.sub(r'\b\d+\b', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Lemmatize words
        words = text.split()
        words = [lemmatizer.lemmatize(
            word) for word in words if word not in domain_stopwords]

        processed_texts.append(' '.join(words))

    return processed_texts


def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """Custom train-test split that ensures all labels in test set are present in training set"""
    # Group by label
    df_combined = pd.DataFrame({'X': X, 'y': y})
    unique_labels = y.unique()
    train_indices = []
    test_indices = []

    for label in unique_labels:
        # Get indices for this label
        label_indices = df_combined[df_combined['y'] == label].index.tolist()

        if len(label_indices) == 1:
            # If only one example, put it in training
            train_indices.extend(label_indices)
        else:
            # Otherwise split according to test_size
            n_test = max(1, int(len(label_indices) * test_size))

            # Shuffle the indices
            np.random.seed(random_state)
            np.random.shuffle(label_indices)

            # Split
            test_indices.extend(label_indices[:n_test])
            train_indices.extend(label_indices[n_test:])

    # Return the split data
    return (X.iloc[train_indices], X.iloc[test_indices],
            y.iloc[train_indices], y.iloc[test_indices])


@timer_decorator
def build_and_train_models(X_train, X_test, y_train, y_test, label_encoder):
    """Build and train multiple models for comparison"""
    print(f"\n{'='*20} MODEL TRAINING AND EVALUATION {'='*20}")

    # Check class distribution
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())

    # Apply SMOTE for imbalanced classes if needed
    counts = Counter(y_train)
    min_samples = min(counts.values())
    if min_samples < 10:
        try:
            print("\nApplying SMOTE to balance classes...")
            smote = SMOTE(random_state=RANDOM_STATE,
                          k_neighbors=min(5, min_samples-1))
            X_train_text = X_train.values.reshape(-1, 1)
            X_train_text, y_train = smote.fit_resample(X_train_text, y_train)
            X_train = pd.Series([x[0] for x in X_train_text])
            print(f"Class distribution after SMOTE: {Counter(y_train)}")
        except Exception as e:
            print(f"Warning: Could not apply SMOTE: {e}")
            print("Proceeding with original imbalanced data.")

    # Common parameters for TF-IDF
    tfidf_params = {
        'max_features': 15000,
        'min_df': 2,
        'max_df': 0.85,
        'ngram_range': (1, 2),
        'sublinear_tf': True
    }

    # Define models to try
    models = {
        'LogisticRegression': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', LogisticRegression(C=10, solver='saga', max_iter=2000,
                                       random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', CalibratedClassifierCV(
                LinearSVC(C=1, random_state=RANDOM_STATE, max_iter=2000),
                cv=5)
             )
        ]),
        'RandomForest': Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=None,
                                           min_samples_split=2, random_state=RANDOM_STATE,
                                           n_jobs=-1))
        ])

    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None

    print("\nTraining and evaluating models with cross-validation:")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        try:
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=RANDOM_STATE)
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

            # Train model on full training set
            model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            # Save results
            train_time = time.time() - start_time
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'training_time': train_time
            }

            print(
                f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Training Time: {train_time:.2f} seconds")

            # Check if this is the best model
            if test_accuracy > best_score:
                best_score = test_accuracy
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"Error training {name}: {e}")

    if not best_model:
        print("All models failed to train. Using LogisticRegression as fallback.")
        best_model = models['LogisticRegression']
        best_model.fit(X_train, y_train)
        best_model_name = 'LogisticRegression'

    # Print summary of results
    if results:
        print("\nModel Performance Summary:")
        summary_df = pd.DataFrame(results).T
        summary_df = summary_df[['cv_mean', 'cv_std',
                                 'test_accuracy', 'training_time']]
        summary_df.columns = [
            'CV Accuracy (Mean)', 'CV Accuracy (Std)', 'Test Accuracy', 'Training Time (s)']
        print(summary_df.sort_values('Test Accuracy', ascending=False))

        # Save performance summary
        summary_df.to_csv('app/reports/model_performance_summary.csv')

    # Detailed evaluation of best model
    print(f"\nDetailed evaluation of best model ({best_model_name}):")
    y_pred = best_model.predict(X_test)

    # Convert encoded y_test and y_pred back to original classes for better readability
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # Get classification report
    print("\nClassification Report:")
    report = classification_report(
        y_test_original, y_pred_original, output_dict=True)
    print(classification_report(y_test_original, y_pred_original))

    # Save report to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('app/reports/classification_report.csv')

    # Plot confusion matrix for top classes
    try:
        top_classes = pd.Series(y_train).value_counts(
        ).index[:min(15, len(pd.Series(y_train).value_counts()))]
        top_class_names = label_encoder.inverse_transform(top_classes)

        # Filter test data for top classes
        mask = np.isin(y_test, top_classes)
        if mask.any():
            y_test_top = y_test[mask]
            y_pred_top = y_pred[mask]

            # Convert to original class names
            y_test_top_names = label_encoder.inverse_transform(y_test_top)
            y_pred_top_names = label_encoder.inverse_transform(y_pred_top)

            # Create confusion matrix
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(
                y_test_top_names, y_pred_top_names, labels=top_class_names)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=top_class_names, yticklabels=top_class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(
                f'Confusion Matrix for Top {len(top_class_names)} Classes')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('app/reports/confusion_matrix.png')
            plt.close()
    except Exception as e:
        print(f"Warning: Could not create confusion matrix: {e}")

    return best_model, results


@timer_decorator
def analyze_feature_importance(best_model, label_encoder):
    """Analyze feature importance for better model understanding"""
    print(f"\n{'='*20} FEATURE IMPORTANCE ANALYSIS {'='*20}")

    try:
        # Extract feature names
        tfidf_vectorizer = best_model.named_steps['tfidf']
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # For tree-based models
        if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
            importances = best_model.named_steps['clf'].feature_importances_

            # Sort features by importance
            indices = np.argsort(importances)[::-1]

            # Print top features
            print("\nTop 30 most important features:")
            feature_importance_data = []
            for i in range(min(30, len(feature_names))):
                feature = feature_names[indices[i]]
                importance = importances[indices[i]]
                print(f"{feature}: {importance:.4f}")
                feature_importance_data.append(
                    {'Feature': feature, 'Importance': importance})

            # Save feature importance data
            pd.DataFrame(feature_importance_data).to_csv(
                'app/reports/feature_importance.csv', index=False)

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.bar(range(min(30, len(indices))),
                    importances[indices[:30]], align="center")
            plt.xticks(range(min(30, len(indices))), [
                       feature_names[i] for i in indices[:30]], rotation=90)
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig('app/reports/feature_importance.png')
            plt.close()

        # For linear models (like LogisticRegression)
        elif hasattr(best_model.named_steps['clf'], 'coef_'):
            # For multi-class problems, analyze coefficients for each class
            if best_model.named_steps['clf'].coef_.shape[0] > 1:
                coef = best_model.named_steps['clf'].coef_
                class_names = label_encoder.classes_

                # For top classes, print and plot most important features
                all_feature_importance = []
                for i, class_name in enumerate(class_names[:min(5, len(class_names))]):
                    print(f"\nMost important features for '{class_name}':")

                    # Get feature importance for this class
                    class_coef = coef[i]
                    top_indices = np.argsort(np.abs(class_coef))[::-1][:15]

                    # Print top features
                    class_feature_importance = []
                    for idx in top_indices:
                        feature = feature_names[idx]
                        importance = class_coef[idx]
                        print(f"{feature}: {importance:.4f}")
                        class_feature_importance.append({
                            'Class': class_name,
                            'Feature': feature,
                            'Importance': importance
                        })
                    all_feature_importance.extend(class_feature_importance)

                # Save all feature importance data
                pd.DataFrame(all_feature_importance).to_csv(
                    'app/reports/feature_importance_by_class.csv', index=False)
            else:
                # For binary classification
                coef = best_model.named_steps['clf'].coef_[0]
                indices = np.argsort(np.abs(coef))[::-1]

                # Print top features
                print("\nTop 30 most important features:")
                feature_importance_data = []
                for i in range(min(30, len(feature_names))):
                    feature = feature_names[indices[i]]
                    importance = coef[indices[i]]
                    print(f"{feature}: {importance:.4f}")
                    feature_importance_data.append(
                        {'Feature': feature, 'Importance': importance})

                # Save feature importance data
                pd.DataFrame(feature_importance_data).to_csv(
                    'app/reports/feature_importance.csv', index=False)

                # Plot feature importance
                plt.figure(figsize=(12, 8))
                plt.bar(range(min(30, len(indices))), np.abs(
                    coef[indices[:30]]), align="center")
                plt.xticks(range(min(30, len(indices))), [
                           feature_names[i] for i in indices[:30]], rotation=90)
                plt.title("Feature Importance (Absolute Coefficient Values)")
                plt.tight_layout()
                plt.savefig('app/reports/feature_importance.png')
                plt.close()
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")


@timer_decorator
def main():
    """Main function to run the job title prediction pipeline"""
    print(f"\n{'='*20} JOB TITLE PREDICTION MODEL TRAINING {'='*20}")

    # 1. Load and explore data
    try:
        df = load_and_explore_data("categorized_jobs.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure 'categorized_jobs.csv' exists and contains 'text' and 'Role' columns.")
        return

    # 2. Preprocess text
    try:
        print("\nPreprocessing text data...")
        df['processed_text'] = preprocess_text(df['Key Skills'])
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        print("Using original text without preprocessing.")
        df['processed_text'] = df['Key Skills']

    # Handle rare job titles - choose one of the approaches below

    # APPROACH 1: Filter out rare roles (recommended for better model performance)
    print("\nHandling rare job titles...")
    min_examples = 5  # Minimum examples per job title
    role_counts = df['Job Title'].value_counts()
    rare_roles = role_counts[role_counts < min_examples].index.tolist()

    df_filtered = df[~df['Job Title'].isin(rare_roles)]
    print(
        f"Removed {len(rare_roles)} rare roles. {len(df_filtered)} samples remaining out of original {len(df)}.")

    # APPROACH 2: Combine rare roles into an "Other" category (uncomment to use)
    # print("\nHandling rare job titles...")
    # min_examples = 5  # Minimum examples per job title
    # role_counts = df['Job Title'].value_counts()
    # rare_roles = role_counts[role_counts < min_examples].index.tolist()
    # df_filtered = df.copy()
    # df_filtered['Job Title'] = df_filtered['Job Title'].apply(lambda x: 'Other' if x in rare_roles else x)
    # print(f"Combined {len(rare_roles)} rare roles into an 'Other' category. {len(df_filtered)} samples total.")

    # 3. Split data
    print("\nSplitting data into train and test sets...")
    X = df_filtered['processed_text']
    y = df_filtered['Job Title']

    try:
        print("Using stratified split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except Exception as e:
        print(f"Error splitting data with stratification: {e}")
        print("Using custom train-test split to ensure all test labels exist in training set...")
        X_train, X_test, y_train, y_test = custom_train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    # 4. Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 5. Train and evaluate models
    try:
        best_model, results = build_and_train_models(
            X_train, X_test, y_train_encoded, y_test_encoded, label_encoder)
    except Exception as e:
        print(f"Error in model training and evaluation: {e}")
        print("Training a simple Logistic Regression model as fallback...")

        # Fallback model
        tfidf = TfidfVectorizer(max_features=10000)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        best_model = Pipeline([('tfidf', tfidf), ('clf', clf)])
        best_model.fit(X_train, y_train_encoded)

    # 6. Analyze feature importance
    try:
        analyze_feature_importance(best_model, label_encoder)
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")

    # 7. Save model and encoder
    print("\nSaving model and encoder...")
    joblib.dump(best_model, "app/ml/career_model.pkl")
    joblib.dump(label_encoder, "app/ml/label_encoder.pkl")

    # Save training metadata
    metadata = {
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_samples': len(df_filtered),
        'num_features': best_model.named_steps['tfidf'].max_features if hasattr(best_model.named_steps['tfidf'], 'max_features') else 'unknown',
        'num_classes': len(label_encoder.classes_),
        'model_type': type(best_model.named_steps['clf']).__name__
    }
    pd.DataFrame([metadata]).to_csv('app/ml/model_metadata.csv', index=False)

    print(f"\n{'='*20} TRAINING COMPLETED {'='*20}")
    print("Model trained and saved successfully to app/ml/career_model.pkl")
    print("Label encoder saved to app/ml/label_encoder.pkl")
    print("Training reports saved to app/reports/")


if __name__ == "__main__":
    main()
