import shap
import matplotlib.pyplot as plt
from data_processing import load_data, optimize_memory, preprocess
from train_model import train_and_select_best_model

def generate_summary_plot():
    print("Training model...")
    
    # Get the model directly from the function — no pkl needed
    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    
    print("Calculating SHAP values...")
    # Create the explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    print("Generating and saving SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    
    # Handle output format depending on model type
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
    
    plt.tight_layout()
    plt.savefig('../shap_summary_plot.png')
    print("Plot saved as shap_summary_plot.png!")
    
    return shap_values, X_test

if _name_ == "_main_":
    generate_summary_plot()