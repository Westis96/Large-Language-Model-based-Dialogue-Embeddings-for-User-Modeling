import data_loader
import language_model_api
import file_writer
from prompt_templates import get_template

def synthesize_data(template_type, num_samples, output_file):
    """
    Generates synthetic data using a language model based on a specified template and personas.
    """
    
    # Load the dataset of personas
    personas_dataset = data_loader.load_personas("path/to/persona/dataset")
    if num_samples > 0:
        personas_dataset = personas_dataset.select(range(num_samples))
    
    # Select the appropriate prompt template
    prompt_template = get_template(template_type)
    if not prompt_template:
        raise ValueError(f"Template '{template_type}' not found.")
        
    # List to hold the generated data
    generated_results = []
    
    # Process each persona in the dataset
    for persona in personas_dataset:
        persona_text = persona.get("text").strip()
        
        if template_type == "dialog":
            # Special handling for generating multi-turn dialogs
            dialogs = generate_dialogs_for_persona(persona_text)
            result = {"input_persona": persona_text, "dialogs": dialogs}
        else:
            # For simpler, single-response templates
            prompt = prompt_template.format(persona=persona_text)
            generated_text = language_model_api.generate_response(prompt)
            result = {"input_persona": persona_text, "synthesized_text": generated_text}
            
        generated_results.append(result)
        
    # Write all the generated results to a JSONL file
    file_writer.write_to_jsonl(output_file, generated_results)
    print(f"Synthetic data generation complete. Results saved to {output_file}.")

def generate_dialogs_for_persona(persona_text, num_dialogs=3, max_turns=4):
    """
    Generates a set of multi-turn dialogs for a given persona.
    """
    all_dialogs = []
    start_template = get_template("dialog_start")
    continue_template = get_template("dialog_continue")
    
    for _ in range(num_dialogs):
        dialog_history = []
        
        # Start the dialog with the persona's first message
        first_prompt = start_template.format(persona=persona_text)
        persona_message = language_model_api.generate_response(first_prompt)
        dialog_history.append({"speaker": "user", "message": persona_message})
        
        # Continue the conversation for a random number of turns
        for _ in range(max_turns - 1):
            # The assistant responds to the user
            assistant_message = language_model_api.generate_conversation_response(dialog_history)
            dialog_history.append({"speaker": "assistant", "message": assistant_message})
            
            # The user (persona) responds to the assistant
            history_str = format_dialog_history(dialog_history)
            next_prompt = continue_template.format(persona=persona_text, history=history_str)
            persona_message = language_model_api.generate_response(next_prompt)
            dialog_history.append({"speaker": "user", "message": persona_message})
            
        all_dialogs.append(dialog_history)
        
    return all_dialogs

def format_dialog_history(dialog):
    """
    Converts a dialog history list into a simple string format.
    """
    return "\n".join([f"{turn['speaker']}: {turn['message']}" for turn in dialog])

def main():
    """
    Main function to configure and run the data synthesis process.
    """
    # Configuration parameters
    template_choice = "instruction"  # e.g., 'instruction', 'knowledge', 'dialog'
    number_of_samples = 50  # 0 for all samples
    output_path = "output/synthesized_data.jsonl"
    
    # Execute the synthesis
    synthesize_data(template_choice, number_of_samples, output_path)

if __name__ == "__main__":
    main() 