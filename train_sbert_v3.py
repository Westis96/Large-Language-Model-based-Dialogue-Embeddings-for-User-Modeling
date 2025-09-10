import model_loader
import data_loader
import model_trainer

def fine_tune_sentence_model(
    base_model_name,
    train_data_path,
    validation_data_path,
    output_directory,
    epochs,
    batch_size,
    learning_rate,
    instruction_prefix,
    input_prefix
):
    """
    Fine-tunes a Sentence Transformer model using a contrastive loss function.
    """
    
    # Load the pre-trained sentence transformer model
    model = model_loader.load_sentence_transformer(base_model_name)
    
    # --- Data Preparation ---
    train_dataset = data_loader.load_dataset(train_data_path)
    validation_dataset = data_loader.load_dataset(validation_data_path) if validation_data_path else None
    
    def format_for_contrastive_loss(dataset):
        return dataset.map_to_pairs(
            instruction_col='instruction', 
            input_col='input', 
            instruction_prefix=instruction_prefix, 
            input_prefix=input_prefix
        )

    formatted_train_data = format_for_contrastive_loss(train_dataset)
    formatted_validation_data = format_for_contrastive_loss(validation_dataset) if validation_dataset else None
    
    training_loss = model_trainer.create_contrastive_loss(model)
    
    training_parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "output_dir": output_directory,
        "evaluation_strategy": "steps" if formatted_validation_data else "no",
        "save_strategy": "steps",
        "logging_steps": 100,
        "save_steps": 500
    }
    
    # --- Model Training ---
    trainer = model_trainer.initialize(
        model=model,
        train_dataset=formatted_train_data,
        eval_dataset=formatted_validation_data,
        loss_function=training_loss,
        training_args=training_parameters
    )
    
    # Start the training process
    print("Starting model fine-tuning...")
    trainer.train()
    print("Training complete.")
    
    # Save the final, fine-tuned model to the specified path
    final_model_path = f"{output_directory}/final_model"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}.")

def main():
    """
    Main function to configure and run the model fine-tuning process.
    """
    
    # Execute the fine-tuning
    fine_tune_sentence_model(
        base_model_name=args.model_name,
        train_data_path=args.train_path,
        validation_data_path=args.validation_path,
        output_directory=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        instruction_prefix=args.instruction_prefix,
        input_prefix=args.input_prefix
    )

if __name__ == "__main__":
    main() 