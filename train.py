import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the classes we just built in the other files
# (Make sure the file names match where you saved these classes!)
from dataset import MultimodalMIMICDataset
from models import VisionEncoder, ClinicalEncoder, CrossAttentionFusion

def main():
    print("Starting the Factory Assembly Line...")

    # --- 1. THE SETUP (Hardware & Hyperparameters) ---
    # Automatically use the GPU if you have one, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Settings for our training run
    batch_size = 16       # How many patients to look at at once
    learning_rate = 1e-4  # How big of a step to take when learning (0.0001)
    num_epochs = 5        # How many times to look through the ENTIRE dataset

    # --- 2. HIRE THE WORKERS (Data & Models) ---
    # Setup the Dataset and DataLoader
    print("Loading data from warehouse...")
    train_dataset = MultimodalMIMICDataset(npy_dir='../data_npy/train', labels_csv_path='../train.csv', split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Bring in our AI models and move them to the GPU
    vision_detective = VisionEncoder(embedding_dim=768).to(device)
    clinical_detective = ClinicalEncoder(input_dim=100, embedding_dim=768).to(device)
    fusion_judge = CrossAttentionFusion(embed_dim=768, num_classes=5).to(device)

    # --- 3. THE RULEBOOK (Loss & Optimizer) ---
    # BCEWithLogitsLoss is the standard for "Multi-Label" classification. 
    # It allows the AI to say "This patient has BOTH Pneumonia AND Edema."
    loss_function = nn.BCEWithLogitsLoss()
    
    # The Optimizer is the manager. It looks at the errors and tweaks the AI's internal math.
    # We pass it the parameters (math weights) of ALL three models at the same time.
    optimizer = optim.Adam(
        list(vision_detective.parameters()) + 
        list(clinical_detective.parameters()) + 
        list(fusion_judge.parameters()), 
        lr=learning_rate
    )

    # --- 4. THE FACTORY FLOOR (The Training Loop) ---
    print("\nStarting Training...")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Enumerate gives us a batch number so we can track progress
        for batch_idx, (images, labs, targets) in enumerate(train_loader):
            
            # Move the data to the GPU
            images = images.to(device)
            labs = labs.to(device)
            targets = targets.to(device)

            # Step A: Clear the manager's memory from the last batch
            optimizer.zero_grad()

            # Step B: The Detectives investigate
            img_embed = vision_detective(images)
            lab_embed = clinical_detective(labs)

            # Step C: The Judge makes a prediction
            predictions = fusion_judge(img_embed, lab_embed)

            # Step D: The Reality Check (Calculate the error)
            loss = loss_function(predictions, targets)

            # Step E: Learn from the mistake (Backpropagation)
            loss.backward()     # The computer figures out exactly WHICH math weights caused the error
            optimizer.step()    # The manager tweaks the dials to fix it

            # Track the average error
            running_loss += loss.item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # End of Epoch Summary
        epoch_loss = running_loss / len(train_loader)
        print(f"=== Epoch {epoch+1} Complete | Average Loss: {epoch_loss:.4f} ===\n")

    print("Training Complete! The models have learned from the data.")

if __name__ == "__main__":
    main()