import data_loader
import embedding_generator
import dimensionality_reducer
import plot_generator

def create_tsne_visualization(
    dataset_path,
    model_name,
    output_directory,
    instruction_prefix,
    input_prefix,
    show_class_colors=False,
    apply_text_formatting=False
):
    """
    Generates a t-SNE plot to visualize instruction and persona embeddings.
    """
    
    # Load the dataset containing text and optional class information
    dataset = data_loader.load(dataset_path)
    
    # --- Text and Embedding Preparation ---
    # Extract instruction and persona texts from the dataset
    instructions = dataset.get_column("instruction")
    personas = dataset.get_column("input")
    
    # Optionally, add prefixes to the texts to simulate formatted model input
    if apply_text_formatting:
        instructions = [f"{instruction_prefix} {text}" for text in instructions]
        personas = [f"{input_prefix} {text}" for text in personas]
        
    # Generate embeddings for both sets of texts using a specified model
    model = embedding_generator.load_model(model_name)
    instruction_embeddings = model.encode(instructions)
    persona_embeddings = model.encode(personas)
    
    # Combine embeddings for joint t-SNE processing
    all_embeddings = combine_embeddings(instruction_embeddings, persona_embeddings)
    
    # --- t-SNE and Plotting ---
    print("Running t-SNE...")
    
    # Apply t-SNE to reduce the embeddings to two dimensions
    tsne_results = dimensionality_reducer.reduce_with_tsne(all_embeddings, dimensions=2)
    
    # Separate the 2D results back into instructions and personas
    tsne_instructions, tsne_personas = split_results(tsne_results, len(instructions))
    
    # Prepare data for plotting, including labels and colors
    plot_data = {
        "x_coords": tsne_results[:, 0],
        "y_coords": tsne_results[:, 1],
        "type": ["Persona"] * len(personas) + ["Instruction"] * len(instructions)
    }
    
    if show_class_colors:
        # If class information is available, add it to the plot data
        instruction_classes = dataset.get_column("instruction_class")
        persona_classes = dataset.get_column("input_class")
        plot_data["class"] = persona_classes + instruction_classes
    
    # Generate the t-SNE scatter plot
    tsne_plot = plot_generator.create_scatterplot_from_data(
        data=plot_data,
        x_col="x_coords",
        y_col="y_coords",
        style_col="type",  # Different markers for personas vs. instructions
        color_col="class" if show_class_colors else None,
        title=f"t-SNE Visualization for {model_name}"
    )
    
    # Save the plot to a file
    plot_generator.save_plot(tsne_plot, output_directory, "tsne_visualization.png")
    
    print(f"t-SNE plot saved to {output_directory}.")

def main():
    """
    Main function to configure and run the t-SNE visualization process.
    """
    
    # Execute the visualization
    create_tsne_visualization(
        dataset_path=args.dataset_path,
        model_name=args.embedding_model,
        instruction_prefix=args.instruction_prefix,
        input_prefix=args.input_prefix,
        output_directory=args.output_dir,
        show_class_colors=True
    )

if __name__ == "__main__":
    main() 