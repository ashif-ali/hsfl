# Hybrid Split Federated Learning

This repository contains Jupyter notebooks for running a simple hybrid split federated learning (HSFL) demo. The server and three clients exchange smashed data and model weights using a Flask API exposed through ngrok.

## Running the server

1. Open `HSFL_Server.ipynb` in Jupyter or Colab and run the cells sequentially.
2. After the Flask application is started, ngrok creates a public URL for port `5000`. The relevant code is:

```python
threading.Thread(target=run_flask).start()
public_url = ngrok.connect(5000).public_url
print("Server URL:", public_url)
```

Running this cell prints a line such as `Server URL: https://xxxx.ngrok-free.app`. Copy this URL; it will be used by every client.

## Updating the clients

Each client notebook (`Client_1.ipynb`, `Client_2.ipynb`, and `Client_3.ipynb`) defines a `server_url` variable:

```python
server_url = 'https://fa07-34-105-36-66.ngrok-free.app/forward_backward'
```

Replace the sample host with the ngrok URL printed by the server while keeping the `/forward_backward` path. Execute the client notebooks after updating this value so they can send requests to the running server.

## Federated averaging

The server aggregates uploaded client weights using the `fed_avg` function:

```python
def fed_avg(weight_files):
    avg_weights = torch.load(weight_files[0])
    for key in avg_weights.keys():
        avg_weights[key] = torch.mean(
            torch.stack([torch.load(wf)[key] for wf in weight_files]), dim=0
        )
    return avg_weights

client_weights = [
    '/content/drive/MyDrive/HSFL_Project/client_1/client_1_weights.pt',
    '/content/drive/MyDrive/HSFL_Project/client_2/client_2_weights.pt',
    '/content/drive/MyDrive/HSFL_Project/client_3/client_3_weights.pt'
]

global_weights = fed_avg(client_weights)
# Save averaged weights explicitly to Drive
torch.save(global_weights, '/content/drive/MyDrive/HSFL_Project/server/global_peripheral_weights.pt')
```

This step loads the weights saved by each client, averages them parameter-wise, and stores the result as the new global model.

## Weight file locations

Clients store their trained weights in:

- `/content/drive/MyDrive/HSFL_Project/client_1/client_1_weights.pt`
- `/content/drive/MyDrive/HSFL_Project/client_2/client_2_weights.pt`
- `/content/drive/MyDrive/HSFL_Project/client_3/client_3_weights.pt`

The aggregated weights produced by the server are written to:

- `/content/drive/MyDrive/HSFL_Project/server/global_peripheral_weights.pt`

Make sure these paths exist in your environment (for example on Google Drive) before running the notebooks.

## Dataset location

Each client expects its training images to be stored under `/content/drive/MyDrive/HSFL_Project/data`. Place the subset of data for each client in
subdirectories named `client_1`, `client_2`, and `client_3` respectively. The notebooks load their local data using PyTorch `DataLoader` instances that
read from these folders.
