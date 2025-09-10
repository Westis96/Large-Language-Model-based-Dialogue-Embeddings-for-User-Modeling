import model_loader
import dataset_loader
import analysis_reporter

def perform_retrieval_analysis(instruction_model_name, input_model_name, dataset_path, output_directory):
    """
    Analyzes how well instruction embeddings can retrieve their corresponding input (persona) embeddings.
    """
    
    # Load the models for generating embeddings
    instruction_model = model_loader.load(instruction_model_name)
    input_model = model_loader.load(input_model_name)
    
    # Load the dataset containing instruction and input texts
    dataset = dataset_loader.load(dataset_path)
    
    instruction_texts = [f"Instruct: {text}" for text in dataset.get_column("instruction")]
    input_texts = [f"Persona: {text}" for text in dataset.get_column("input")]
    
    # Generate embeddings for all instructions and inputs
    instruction_embeddings = instruction_model.encode(instruction_texts)
    input_embeddings = input_model.encode(input_texts)
    
    # Calculate cosine similarity between each instruction and all inputs
    similarity_scores = calculate_cosine_similarity(instruction_embeddings, input_embeddings)
    
    # Determine the rank of the correct input for each instruction
    ranks = []
    for i, instruction_embedding in enumerate(instruction_embeddings):
        # The correct input is at the same index 'i' in the input_embeddings list
        correct_input_index = i
        
        # Get similarity scores of the current instruction against all inputs
        scores_for_instruction = similarity_scores[i]
        
        # Sort inputs by similarity score in descending order and find the rank of the correct input
        sorted_indices = sort_indices_by_score(scores_for_instruction, descending=True)
        rank = find_position(sorted_indices, correct_input_index) + 1  # +1 because rank is 1-based
        ranks.append(rank)
        
    # Calculate key performance metrics
    top_1_accuracy = calculate_accuracy(ranks, top_k=1)
    top_5_accuracy = calculate_accuracy(ranks, top_k=5)
    mean_reciprocal_rank = calculate_mrr(ranks)
    
    # Prepare the results for reporting
    results = {
        "Top-1 Accuracy": top_1_accuracy,
        "Top-5 Accuracy": top_5_accuracy,
        "Mean Reciprocal Rank (MRR)": mean_reciprocal_rank,
        "model_info": {
            "instruction_model": instruction_model_name,
            "input_model": input_model_name,
            "dataset": dataset_path
        }
    }
    
    # Save the analysis results to a file
    analysis_reporter.save_results(results, output_directory, "retrieval_analysis_summary.json")
    
    print("Retrieval analysis complete. Results saved.")

def main():
    """
    Main execution block to run the retrieval analysis with predefined parameters.
    """
    # Configuration for the analysis
    instruction_model = "path/to/instruction_model"
    input_model = "path/to/input_model"
    dataset_location = "path/to/dataset"
    results_directory = "results/retrieval_analysis"
    
    # Run the analysis
    perform_retrieval_analysis(instruction_model, input_model, dataset_location, results_directory)

if __name__ == "__main__":
    main() 