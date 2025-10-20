import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from model import CoolUncoolCNN, CoolUncoolTrainer, CoolUncoolDataset, get_data_transforms

def load_labeled_data(image_directory):
    labels_file = os.path.join(image_directory, "labels.json")
    
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    image_paths = []
    label_list = []
    
    for filename, label in labels.items():
        image_path = os.path.join(image_directory, filename)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            label_list.append(label)
    
    return image_paths, label_list

def create_data_loaders(image_paths, labels, batch_size=16, train_split=0.8):
    train_transform, val_transform = get_data_transforms()
    
    dataset = CoolUncoolDataset(image_paths, labels, transform=train_transform)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = '/Users/jovgav/Desktop/MLClass/vogue_images'
    print(f"Data directory: {data_dir}")
    
    try:
        image_paths, labels = load_labeled_data(data_dir)
        print(f"Loaded {len(image_paths)} labeled images")
        
        cool_count = sum(1 for l in labels if l == 1)
        uncool_count = sum(1 for l in labels if l == 0)
        print(f"Cool images: {cool_count}")
        print(f"Uncool images: {uncool_count}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the labeling app first to create labels.json")
        return
    
    train_loader, val_loader = create_data_loaders(image_paths, labels)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    model = CoolUncoolCNN(num_classes=2)
    trainer = CoolUncoolTrainer(model, device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    print(f"\nStarting training for 50 epochs...")
    train_losses, train_accs, val_losses, val_accs = trainer.train(
        train_loader, val_loader, epochs=50
    )
    
    model_path = os.path.join(data_dir, "cool_uncool_model.pth")
    trainer.save_model(model_path)
    
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, val_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {accuracy:.2%}")
    print(f"Model saved to: {model_path}")
    print("Training complete!")

if __name__ == "__main__":
    main()