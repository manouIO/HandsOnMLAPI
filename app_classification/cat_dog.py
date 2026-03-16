from datasets import load_dataset
# Now you can load your dataset
dataset=  load_dataset("AIOmarRehan/AnimalsDataset")
print("SUCCESS!")
print(dataset)



# View the first row
print(dataset['train'][0])

# If it's an image dataset, you can display the image directly in VS Code
dataset['train'][0]['image']