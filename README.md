# Collaborative Computation using Federated Learning
This is my Honours year research project on federated learning with my supervisor Prof Amanda Barnard and in collaboration with Choiceflows. 

# Goal
The goal of this project is to provide a Python package in federated learning (FL) that's compatiple with existing machine learning (ML) packages in Python (currently I'm using PyTorch).

# Use
Currently the package is under development. However, demonstration of the package in a local environment can be run as follows:
- Download the package
  <br>
- Install dependencies using requirements.txt file in your environment using
  ```
  pip install -r requirements.txt
  ```
  <br>
- Prepare your datasets (in this case, we assume there are 8 datasets in .csv format). Your repository should look like this
  ```bash
  -- datasets -- 1.csv
  |          |-- 2.csv
  |          |-- 3.csv
  |          ......
  |
  -- fl -- Federator.py
       |-- Clients.py
       |-- Model.py
       ......
  ```
  <br>
- Open up a new terminal in your environment (local base or venv) and use the following command to start a Federator:
  ```
  python Federator.py --h 127.0.0.1 --p 65432 --n 8 --rounds 2 --ratio 0.85 --x 1 --e 5
  ```
  This will start a Federator with expectation of 8 client connections on localhost (127.0.0.1) port 65432. <br>
  <br>
  There will be 2 communication rounds with each client (i.e., every client will report twice about the model parameters to the Federator). <br>
  <br>
  The minimal explain ratio at the dimensionality reduction step is 85%. And there will have encryption for the communication (--x 1). Each time a client would train 5 epochs using their data. <br>
- Open up another terminal in the same environment and use the following command to start the 8 clients:
  ```
  python StartClients.py
  ```
  This will start 8 clients to perform the learning task. Note that if there are less than 8 datasets in your "dataset" folder, this would still run, but the Federator would NOT be able to proceed. <br>
  <br>
  To solve this, you need to change the --n parameter in the previous command to the right number of clients and restart the Federator. <br>
- When the training finishes, you should see 8 folders named client0-7 under your fl folder and the Federator would continue to run in your first terminal.
  Your repository structure should now look like this:
  ```bash
  -- datasets -- 1.csv
  |          |-- 2.csv
  |          |-- 3.csv
  |          ......
  |
  -- fl -- Federator.py
       |-- Clients.py
       |-- Model.py
       |-- client0 -- ......
       |-- client1 -- ......
       ......
  ```
