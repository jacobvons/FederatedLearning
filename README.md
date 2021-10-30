# Collaborative Computation using Federated Learning
## Acknowledgement
This is my Honours year (2021) research project on federated learning with my supervisor [Prof Amanda Barnard](https://cs.anu.edu.au/people/amanda-barnard) at the [ANU](https://www.anu.edu.au).

## Goal
The goal of this project is to develop a general Python package for federated learning (FL) for tabular data that's compatiple with existing machine learning (ML) packages in Python (currently using PyTorch). Integreting PCA into the regular FL framework as a means of data preprocessing is also expected. 

## Dataset
The datasets consist of eight public available synthetic nanoparticle datasets. References to them are listed below:
- Barnard, Amanda; Opletal, George (2019): Disordered Silver Nanoparticle Data Set. v1. CSIRO. Data Collection. https://doi.org/10.25919/5e30b5231c669
- Barnard, Amanda; Sun, Baichuan; Motevalli Soumehsaraei, Benyamin; Opletal, George (2017): Silver Nanoparticle Data Set. v3. CSIRO. Data Collection. https://doi.org/10.25919/5d22d20bc543e
- Barnard, Amanda; Opletal, George (2019): Gold Nanoparticle Data Set. v1. CSIRO. Data Collection. https://doi.org/10.25919/5d395ef9a4291
- Barnard, Amanda; Opletal, George (2019): Palladium Nanoparticle Data Set. v1. CSIRO. Data Collection. https://doi.org/10.25919/5d3958ee6f239
- Barnard, Amanda; Sun, Baichuan; Opletal, George (2018): Platinum Nanoparticle Data Set. v2. CSIRO. Data Collection. https://doi.org/10.25919/5d3958d9bf5f7

Brief description of the datasets:
![Dataset Description](https://github.com/jacobvons/FederatedLearning/blob/main/dataDes.jpg?raw=true)


## To Use
Currently the package is under development and not tested in an actual server environment. However, demonstration of the package in a local environment can be run in three ways as follows:

### Use ```StartClients.py```
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
  python Federator.py --h 127.0.0.1 --p 65432 --n 8 --rounds 2 --ratio 0.85 --x 1 --e 5 --name mse_loss
  ```
  This will start a Federator with expectation of 8 client connections on localhost (127.0.0.1) port 65432. <br>
  <br>
  There will be 2 communication rounds with each client (i.e., every client will report twice about the model parameters to the Federator). <br>
  <br>
  The minimal explain ratio at the dimensionality reduction step is 85%. And there will have encryption for the communication (```--x 1```). Each time a client would train 5 epochs using their data. <br>
- Open up another terminal in the same environment and use the following command to start the 8 clients:
  ```
  python StartClients.py
  ```
  This will start 8 clients to perform the learning task. Note that if there are less than 8 datasets in your "dataset" folder, this would still run, but the Federator would NOT be able to proceed. <br>
  <br>
  To solve this, you need to change the ```--n``` parameter in the previous command to the right number of clients and restart the Federator. <br>
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
       |-- mse_loss ----- client0
       |              |-- client1 -- -- ......
       |              |             |-- ......
       |              |             |-- ......
       |              |             ......
       |              |-- client2 -- ......
       |              ......
       ......
  ```

### Use ```Client.py```
- Similar to using ```StartClient.py```, ```Client.py``` is dedicated at creating single clients one per process. If you inspect the source code, you'll see StartClients.py is just using a for loop to create clients same way as ```Client.py``` would one by one. This is for the sake of easy testing at my testing stage for all my eight datasets. And in an actual distributed environment, it would only make sense that each client release their job separately using ```Client.py``` over ```StartClients.py```. 
- After you've downloaded the package and prepared your dataset in the same way above, you can start the Federator using the same command above.
- Depending on what you input for the ```--n``` parameter, in that many separate terminals, you start clients one at a time using a new terminal using this command: ```python Client.py --h 127.0.0.1 --p 65432 --path path_to_dataset --i client_id```
- After you've created the expected amount of clients, the process will begin and you should see the same behaviour as before using ```StartClients.py```.

### Use ```autotest.py```
- ```autotest.py``` is a scripted developed for easy and convenient testing. The mechanism is to automatically generate subprocesses for both the federator and the clients using the ```StartClients.py``` logics. 
- It also allows users to put sets of parameters into a .csv file so that the program automatically tests on each of them in order. This is convenient for bulk testing. 
- To use it, first, simply create a .csv file containing the argument sets you want to test. Then, change the testing dataset path in the ```StartClients.py``` file. Finally, run ```python autotest.py``` in your terminal. 
- Refer to the ```sample_test_args.csv``` file for detailed parameter explanation. 

## Customisation
- The system supports customisation from several aspects. One can design their own machine learning models, loss functions, and decide which one to use easily. Simply follow the convention in the ```Model.py``` and ```Loss.py``` to customise models for your own usages. 
- To use them, simply replace the model and loss function in the ```Federator.py``` file. 
- Currently, model and loss function swithing functionality cannnot be achieved directly through test argument .csv file. This is expected to be implemented in later versions. 
- As examples, we have implemented several models and loss functions in those two files using PyTorch. 

## Hyperparameter Tuning
- The system supports hyperparameter tuning mechanism through a Python decorator (```hyper_tune``` implemented in ```Decorator.py```). This decorator is implemented to be compatible with the ```train_epochs``` function in ```Client.py``` and ```Centralised.py```. 
- To use it, simply put "@hyper_tune" above the definition of the ```train_epochs``` function. 
- To customise, one could implement functions with same input and output format as the ```train_epochs``` function and use the decorator directly. 
- The hyperparameter configurations are read through a file users can put in the directory. See ```sample_hyper_config.csv``` for more information. 
