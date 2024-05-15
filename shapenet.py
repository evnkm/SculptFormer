import os
import random
import shutil
import argparse

def random_sample_png(base_path, destination_path):
    # List of categories
    categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # List to store full paths of selected images
    selected_image_paths = []

    # Ensure the destination directory exists, create if not
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Loop through each category
    for category in categories:
        category_path = os.path.join(base_path, category)
        objects = [o for o in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, o))]
        
        # Randomly sample 5 objects if there are enough objects, otherwise sample all available
        sampled_objects = random.sample(objects, min(5, len(objects)))

        # For each sampled object, select a random PNG image
        for obj in sampled_objects:
            obj_path = os.path.join(category_path, obj, 'rendering')
            png_files = [f for f in os.listdir(obj_path) if f.endswith('.png')]
            selected_png = random.choice(png_files)
            full_path = os.path.join(obj_path, selected_png)
            selected_image_paths.append(full_path)

            # Create a new file name using category and object id
            new_file_name = f"{category}.{obj}.{selected_png}"
            destination_file_path = os.path.join(destination_path, new_file_name)
            
            # Copy the file with the new name
            shutil.copy(full_path, destination_file_path)
    
    # Write the source paths to a text file in the destination directory
    source_paths_file = os.path.join(destination_path, 'source_paths.txt')
    with open(source_paths_file, 'w') as f:
        for path in selected_image_paths:
            f.write(path + '\n')
    
    return selected_image_paths

def main():
    parser = argparse.ArgumentParser(description='Copy sampled images from ShapeNet to a specified directory with new names.')
    parser.add_argument('base_path', type=str, help='The base path to the ShapeNet data directory.')
    parser.add_argument('destination_path', type=str, help='The destination path where images should be copied and renamed.')
    
    args = parser.parse_args()

    # Run the function
    selected_images = random_sample_png(args.base_path, args.destination_path)

    # Print selected images and their new location
    for image_path in selected_images:
        new_file_name = os.path.basename(image_path)
        print(f"Copied and renamed to: {os.path.join(args.destination_path, new_file_name)}")

if __name__ == '__main__':
    main()
