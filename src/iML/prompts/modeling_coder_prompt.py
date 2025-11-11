# src/iML/prompts/modeling_coder_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt
from ..utils.utils import smart_truncate_error

class ModelingCoderPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for modeling.
    """

    def default_template(self) -> str:
        """Default template to request LLM to generate modeling code."""
        return """
You are an expert ML engineer. Your task is to generate a COMPLETE and EXECUTABLE Python script for modeling.
This script will be combined with the provided preprocessing code.

## CONTEXT
- **Dataset Name**: {dataset_name}
- **Task Description**: {task_desc}
- **File Paths**: {file_paths} (LOAD DATA FROM THESE PATHS)
- **Data File Description**: {data_file_description}
- **Output data format**: {output_data_format}
- **Submission file description**: {submission_file_description}

{data_handling_instruction}
## MODELING GUIDELINES:
{modeling_guideline}


## PREPROCESSING CODE (Do NOT include this in your response):
The following preprocessing code, including a function `preprocess_data(file_paths: dict)`, will be available in the execution environment. You must call it to get the data.
```python
{preprocessing_code}
```

## REQUIREMENTS:
1.  **Generate COMPLETE Python code for the modeling part ONLY.** Do NOT repeat the preprocessing code.
2.  Your code should start with necessary imports for modeling.
3.  Define an appropriate `train_and_predict()` function based on the data handling approach.
4.  The main execution block (`if __name__ == "__main__":`) must:
    a. Call preprocess_data() to get the data.
    b. **Handle the data based on the iteration type (see IMPORTANT DATA HANDLING).**
    c. Call your train_and_predict() function.
    d. Save the final predictions to a submission.csv file.
5.  **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error to stderr and **exit with a non-zero status code** (`sys.exit(1)`).
6.  Follow the modeling guidelines for algorithm choice.
7.  Do not use extensive hyperparameter tuning unless specified. Keep the code efficient.
8.  Limit comments in the code.
9.  **CRITICAL - ID FORMAT IN SUBMISSION**: Check the modeling guidelines for the field 'IDs_in_submission_file_contain_file_extensions'. If TRUE, include file extensions in IDs (e.g., 'image001.jpg'). If FALSE, use IDs without extensions (e.g., 'image001'). Ensure ID format matches exactly what's expected in the submission file.
10. The submission file must have the same structure (number of columns) as the sample submission file provided in the dataset, but may have different ID. You have to use the test data to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
11. Your final COMPLETE Python code should have only ONE main function. If there are duplicate main function, remove the duplicates and keep only one main function.
12. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
13. **REPRODUCIBILITY & VALIDATION**: 
    - Always use random_state=42 for ALL random operations (model initialization, etc.)
    - Use the validation set for model evaluation (no cross-validation needed)
    - Print the validation score in the format: "Validation Score: <score_value>"
    - Use the evaluation metrics specified in the modeling guidelines
14. Pay attention to the create_submission guideline in the modeling guidelines to create valid submission, for example, if the guideline says that '1_2_1' for image 1, row 2, column 1, the index of row and column should begin with 1, not 0.
"""

    def build(self, guideline: Dict, description: Dict, preprocessing_code: str, previous_code: str = None, error_message: str = None, iteration_type: str = None) -> str:
        """Build prompt to generate modeling code."""
        
        modeling_guideline = guideline.get('modeling', {})
        
        # Add iteration-specific modeling guidance
        iteration_guidance = self._get_iteration_guidance(iteration_type)
        enhanced_guideline = json.dumps(modeling_guideline, indent=2)
        if iteration_guidance:
            enhanced_guideline += f"\n\n## ITERATION-SPECIFIC GUIDANCE:\n{iteration_guidance}"

        # Get data handling instruction based on iteration type
        data_handling = self._get_data_handling_instruction(iteration_type)

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            data_file_description=description.get('data file description', 'N/A'),
            output_data_format=description.get('output_data', 'N/A'),
            modeling_guideline=enhanced_guideline,
            preprocessing_code=preprocessing_code,
            data_handling_instruction=data_handling,
            submission_file_description=description.get('submission file description', 'N/A')
        )

        # Append full description analysis as JSON context
        try:
            description_json = json.dumps(description, indent=2, ensure_ascii=False)
        except Exception:
            description_json = json.dumps(description, indent=2)
        prompt += f"\n\n## FULL DESCRIPTION ANALYSIS (JSON)\n```json\n{description_json}\n```\n"

        if previous_code and error_message:
            # Use smart truncation for error message to save tokens and focus on relevant parts
            max_lines = getattr(self.manager.config, 'max_error_lines_for_llm', 20)
            max_chars = getattr(self.manager.config, 'max_error_message_length', 2048)
            truncated_error = smart_truncate_error(error_message, max_lines=max_lines, max_chars=max_chars)
            
            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The previously generated code failed with an error.

### Previous Code:
```python
{previous_code}
```

### Error Message:
```
{truncated_error}
```

## FIX INSTRUCTIONS:
1. Analyze the error message and the previous code carefully.
2. Fix the specific issue that caused the error.
3. Ensure your code correctly uses the data returned by the `preprocess_data` function.
4. Generate a new, complete, and corrected version of the Python code that resolves the issue.
5. Adhere to all original requirements.

Generate the corrected Python code:
"""
            prompt += retry_context
        
        self.manager.save_and_log_states(prompt, "modeling_coder_prompt.txt")
        return prompt
    
    def _get_iteration_guidance(self, iteration_type: str = None) -> str:
        """Get iteration-specific modeling guidance."""
        if iteration_type == "traditional":
            return """
For Traditional ML algorithms:
- Focus on feature importance analysis
- Try to extract features from image or text data if neccessary
- Use early stopping for gradient boosting methods
- Optimize for tabular data characteristics
- #IMPORTANT: YOU MUST use optuna library for hyperparameter tuning, limit the time of the hypertuning to 3600 second.
"""
        elif iteration_type == "custom_nn":
            return """
For Custom Neural Networks:
- Design NN architecture from scratch using PyTorch.
- Include proper layer definitions (Dense, Dropout, BatchNormalization)
- Implement training loop with validation monitoring
- Use appropriate loss functions and optimizers (Adam, SGD)
- Add learning rate scheduling and early stopping
- Include model checkpointing for best weights
- Handle overfitting with regularization techniques
- Monitor training/validation loss curves
- Use large number of epochs with early stopping = 3
"""
        elif iteration_type == "custom_nn_search":
            return """
For Custom Neural Networks with Architecture Search:
- Adapt the suggested architecture pattern from the guideline to fit this specific problem
- Use PyTorch to implement the architecture with appropriate layer types and connections
- Adjust layer sizes, number of layers, and hyperparameters based on data characteristics
- Include proper layer definitions (Dense, Dropout, BatchNormalization as appropriate)
- Implement training loop with validation monitoring
- Use appropriate loss functions and optimizers (Adam, SGD)
- Add learning rate scheduling and early stopping
- Include model checkpointing for best weights
- Handle overfitting with regularization techniques
- Monitor training/validation loss curves
- Use large number of epochs with early stopping = 3
"""
        elif iteration_type == "pretrained":
            return """
For Pretrained Models (prioritize PyTorch):
- Load and fine-tune pretrained models (PyTorch backend preferred)
- Use transformers library with PyTorch backend for text models
- Use torchvision for vision models (ResNet, ViT, etc.)
- Implement transfer learning approach with proper layer freezing/unfreezing
- Use HuggingFace tokenizers and PyTorch data loaders
- Fine-tune with appropriate learning rates (often lower than training from scratch)
- Handle model adaptation for target task (classification head modification)
- Use PyTorch-specific optimizers and schedulers
- Implement gradual unfreezing strategy if needed
- Prefer torch.nn.functional and PyTorch ecosystem
- Use large number of epochs with early stopping = 3
"""
        else:
            return ""
    
    def _get_data_handling_instruction(self, iteration_type: str = None) -> str:
        """Get data handling instruction based on iteration type."""
        if iteration_type == "traditional":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **preprocessed DataFrames/arrays** (e.g., X_train, X_val, X_test, y_train, y_val, y_test) that are already loaded into memory. Your code should directly use these preprocessed datasets.

### Data Usage:
- **For traditional ML**: Use the preprocessed DataFrames/arrays directly with scikit-learn models
- **Logic**: Call `X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_paths)` and use them directly
- **Memory**: Data is already loaded into memory and ready for training

## CODE STRUCTURE EXAMPLE:
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

def train_and_predict(X_train, y_train, X_val, y_val, X_test):
    # Initialize traditional ML model
    # TODO: Replace below lines with the appropriate modeling code based on guidelines
    model = RandomForestClassifier(random_state=42, n_estimators=100)  # Example model
    
    # Train the model on full training data
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_predictions = model.predict(X_val)
    val_score = accuracy_score(y_val, val_predictions)
    print(f"Validation Score: {{val_score:.4f}}")
    
    # Make predictions on test set
    print("Making predictions on test set...")
    predictions = model.predict(X_test)
    
    return predictions

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # Get preprocessed data (traditional approach returns arrays/DataFrames)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_paths)
        print("Data loaded successfully.")
        
        # Extract test IDs (assuming they're included in preprocessing or index)
        test_ids = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
        
        # Train and predict
        predictions = train_and_predict(X_train, y_train, X_val, y_val, X_test)
        
        # Create submission
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)
        
        print("Modeling script executed successfully!")
        
    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```"""
        
        elif iteration_type == "custom_nn":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of generators** (e.g., `train_gen, val_gen, test_gen`) to save memory. Your code must handle these generators efficiently for neural network training.

### Batch Training for Neural Networks:
- **For custom NN**: Use the generators in training loops with proper batching
- **Logic**: Iterate through generators, get batches, and train the neural network incrementally
- **Memory Efficiency**: This approach keeps memory usage low by processing data in batches

## CODE STRUCTURE EXAMPLE:
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import sys


class CustomNN(nn.Module):
    # Define your custom neural network architecture here

def train_and_predict(train_gen, val_gen, test_gen):
    # Model will be initialized after seeing first batch
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop
    print("Starting neural network training...")
    # Train your custom neural network with appropriate code.
    # Example: Initialize model, train in epochs with batches
    
    # Validation evaluation
    print("Evaluating on validation set...")
    val_predictions = []  # Replace with actual validation predictions
    val_true = []  # Replace with actual validation labels
    # Example: Evaluate model on validation set using val_gen
    
    val_score = accuracy_score(val_true, val_predictions)  # Replace with appropriate scoring metric
    print(f"Validation Score: {{val_score:.4f}}")
    
    # Test predictions
    print("Making predictions on test set...")
    test_predictions = []  # Replace with actual test predictions
    test_ids = []  # Replace with actual test IDs
    # Example: Generate predictions on test set using test_gen
    
    return test_predictions, test_ids

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # Get data generators
        train_gen, val_gen, test_gen = preprocess_data(file_paths)
        print("Data generators initialized successfully.")
        
        # Train and predict
        predictions, test_ids = train_and_predict(train_gen, val_gen, test_gen)
        
        # Create submission
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)
        
        print("Modeling script executed successfully!")
        
    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```"""
        
        elif iteration_type == "pretrained":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of PyTorch DataLoader objects** (e.g., `train_loader, val_loader, test_loader`) with FINITE length formatted for pretrained models.

### Pretrained Model Data Handling (PyTorch):
- **For pretrained models**: Use the provided PyTorch DataLoader objects directly
- **CRITICAL**: DataLoaders have FINITE length (len(train_loader) returns number of batches). Your training loop MUST iterate exactly once per epoch.
- **Training Logic**: Use standard PyTorch training pattern:
  ```python
  for epoch in range(num_epochs):
      for batch_idx, batch in enumerate(train_loader):
          # Training step
          # This loop will STOP after len(train_loader) batches
  ```
- **DO NOT**: Create infinite loops or IterableDataset without proper stopping conditions
- **PyTorch Integration**: Use torch.nn.Module, torch.optim, standard training loops
- **HuggingFace**: Prefer transformers library with PyTorch backend
- **Validation**: Iterate through val_loader ONCE per validation check
- **Testing**: Iterate through test_loader ONCE for final predictions

## CODE STRUCTURE EXAMPLE:
```python
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score
import sys

def train_and_predict(train_loader, val_loader, test_loader):
    # CRITICAL: train_loader, val_loader, test_loader are PyTorch DataLoader with FINITE length
    # len(train_loader) gives number of batches
    
    # Load pretrained model (adjust based on your task type)
    # TODO: Replace "pretrained_model_name" with actual model name based on guideline
    model_name = "pretrained_model_name" 
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    
    # Training loop with FINITE iterations
    print(f"Training for {{num_epochs}} epochs, {{len(train_loader)}} batches per epoch...")
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # This loop will STOP after len(train_loader) batches
            # Extract features and labels from batch
            # TODO: Adjust based on your data format
            inputs, labels = batch  # Example unpacking
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass and training
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {{epoch+1}}/{{num_epochs}}, Batch {{batch_idx}}/{{len(train_loader)}}, Loss: {{loss.item():.4f}}")
        
        print(f"Epoch {{epoch+1}} completed.")
    
    # Validation - iterate ONCE through val_loader
    print("Evaluating on validation set...")
    model.eval()
    val_predictions = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            # This loop will STOP after len(val_loader) batches
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            val_predictions.extend(preds.cpu().numpy())
            val_true.extend(labels.numpy())
    
    val_score = accuracy_score(val_true, val_predictions)
    print(f"Validation Score: {{val_score:.4f}}")
    
    # Test predictions - iterate ONCE through test_loader
    print("Making predictions on test set...")
    test_predictions = []
    test_ids = []
    with torch.no_grad():
        for batch in test_loader:
            # This loop will STOP after len(test_loader) batches
            # TODO: Adjust based on your data format
            inputs, ids = batch  # Test loader may return inputs and IDs
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_ids.extend(ids)
    
    return test_predictions, test_ids

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # Get PyTorch DataLoaders (FINITE length)
        train_loader, val_loader, test_loader = preprocess_data(file_paths)
        print(f"Data loaders initialized: train={{len(train_loader)}} batches, val={{len(val_loader)}} batches, test={{len(test_loader)}} batches")
        
        # Train and predict
        predictions, test_ids = train_and_predict(train_loader, val_loader, test_loader)
        
        # Create submission
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)
        
        print("Modeling script executed successfully!")
        
    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```"""
        
        else:
            # Default behavior (fallback)
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns data based on the model type. Handle the data efficiently according to your chosen algorithm.

### General Guidelines:
- **For traditional ML**: Use DataFrames/arrays directly (if provided)
- **For neural networks**: Use generators for memory efficiency (if provided)
- **Validation**: Always use the validation set for model evaluation
- **Consistency**: Ensure data handling matches your algorithm's requirements"""

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "modeling_code_response.py")
        return code
