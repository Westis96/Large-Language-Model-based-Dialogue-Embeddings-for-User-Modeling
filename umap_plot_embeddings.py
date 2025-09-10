import data_loader
import embedding_generator
import dimensionality_reducer
import plot_generator

def create_umap_visualization(
    dataset_path,
    model_name,
    output_directory,
    show_class_colors=False,
    apply_text_formatting=False
):
    """
    Generates a UMAP plot to visualize instruction and persona embeddings.
    """
    
    # Load the dataset with text and optional class information
    dataset = data_loader.load(dataset_path)
    
    # --- Text and Embedding Preparation ---
    # Extract texts for instructions and personas
    instructions = dataset.get_column("instruction")
    personas = dataset.get_column("input")
    
    # Optionally format texts with prefixes
    if apply_text_formatting:
        instructions = [f"Query: {text}" for text in instructions]
        personas = [f"Persona: {text}" for text in personas]
        
    # Generate embeddings using the specified model
    model = embedding_generator.load_model(model_name)
    instruction_embeddings = model.encode(instructions)
    persona_embeddings = model.encode(personas)
    
    # Combine embeddings for UMAP processing
    all_embeddings = combine_embeddings(instruction_embeddings, persona_embeddings)
    
    # --- UMAP and Plotting ---
    print("Running UMAP...")
    
    # Apply UMAP to reduce embeddings to two dimensions
    umap_results = dimensionality_reducer.reduce_with_umap(all_embeddings, dimensions=2)
    
    # Prepare data for plotting
    plot_data = {
        "x_coords": umap_results[:, 0],
        "y_coords": umap_results[:, 1],
        "type": ["Persona"] * len(personas) + ["Instruction"] * len(instructions)
    }
    
    if show_class_colors:
        # Add class information if available for color-coding
        instruction_classes = dataset.get_column("instruction_class")
        persona_classes = dataset.get_column("input_class")
        plot_data["class"] = persona_classes + instruction_classes
    
    # Create the UMAP scatter plot
    umap_plot = plot_generator.create_scatterplot_from_data(
        data=plot_data,
        x_col="x_coords",
        y_col="y_coords",
        style_col="type",  # Use different markers for types
        color_col="class" if show_class_colors else None,
        title=f"UMAP Visualization for {model_name}"
    )
    
    # Save the plot
    plot_generator.save_plot(umap_plot, output_directory, "umap_visualization.png")
    
    print(f"UMAP plot saved to {output_directory}.")

def main():
    """
    Main function to run the UMAP visualization process.
    """
    # Configuration
    dataset_file_path = "path/to/classified/dataset"
    embedding_model = "some-sentence-transformer-model"
    output_dir = "results/umap_plots"
    
    # Execute visualization
    create_umap_visualization(
        dataset_path=dataset_file_path,
        model_name=embedding_model,
        output_directory=output_dir,
        show_class_colors=True
    )

if __name__ == "__main__":
    main() 