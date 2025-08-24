from jaxtyping import Float
import torch
from einops import rearrange
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import glob
import pandas as pd
from functools import reduce
from models import cElegansFwdSAE, cElegansBwdSAE, cElegansSAE


def load_data() -> Float[torch.Tensor, "worms neurons timesteps"]:
    # Load all CSV files from data directory
    csv_files = glob.glob("data/*.csv")

    # Read each CSV and process
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Find the 'neuron' column index
        neuron_idx = df.columns.get_loc("neuron")
        # Take all columns after 'neuron'
        df = df.iloc[:, neuron_idx:]
        dfs.append(df)

    # Merge all dataframes on common neurons

    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], start=1):
        # use i-1 for the left side, i for the right
        merged_df = pd.merge(
            merged_df, df, on="neuron", how="outer", suffixes=("", f"_{i}")
        )

    # Drop columns that don't end with 's' (except 'neuron')
    cols_to_keep = [
        col
        for col in merged_df.columns
        if col == "neuron" or col.split("_")[0].endswith("s")
    ]
    merged_df = merged_df[cols_to_keep]

    print(merged_df.shape)

    # Convert to torch tensor and transpose to get shape [worms, neurons, timesteps]
    # Drop the 'neuron' column and convert to numpy array
    data_array = merged_df.drop("neuron", axis=1).to_numpy()

    # Convert to torch tensor
    data = torch.tensor(data_array, dtype=torch.float32)

    return data


if __name__ == "__main__":
    data = load_data()
    data = rearrange(data, "neurons timesteps -> timesteps neurons")

    # Check for NaN values in the data
    if torch.isnan(data).any():
        print("Warning: Dataset contains NaN values!")
        print(f"Number of NaN values: {torch.isnan(data).sum().item()}")

    # Remove timesteps that contain any NaN values
    nan_mask = ~torch.isnan(data).any(dim=1)  # Find timesteps without NaNs
    data = data[nan_mask]
    print(f"Shape after removing NaN timesteps: {data.shape}")

    encode = cElegansFwdSAE()
    decode = cElegansBwdSAE()
    batch_size = 1024
    epochs = 100
    sparsity_weight = 5

    # Create DataLoader for batch processing
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Create cosine learning rate scheduler
    optimizer = torch.optim.Adam([*encode.parameters(), *decode.parameters()], lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs,  # Maximum number of epochs
    #     eta_min=1e-6,  # Minimum learning rate
    # )
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Extract tensor from batch tuple
            batch = batch[0]

            hidden = encode(batch)
            hidden = torch.relu(hidden)
            output = decode(hidden)

            loss1 = loss_fn(output, batch)
            loss2 = torch.norm(hidden, p=1) / hidden.numel()

            non_zero = (hidden != 0).sum()

            loss = loss1 + sparsity_weight * loss2

            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch} reconstruction: {loss1.item() / batch_size}, sparsity: {loss2.item() / batch_size}, non_zero: {non_zero.item() / batch_size}"
        )
