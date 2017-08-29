# HISI

Please find in this repository the full code implementation of the project. 
The HISI algorithm aims to segment object of an image. The project also implements a CNN to train and test the result of the HISI algorithm on occluded images. 

For further information of the project, a presentation pdf is available. For more details on the different steps done by the HISI algorithm (initially called Fast Laminart), my master project is available. For details of the implementation, please refer to the code itself. 

To run the HISI algorithm and see its results, please run the code: run_HISI.py using python 3. Several images examples are given in the code. 

To run the entire simulation on the MNIST database, the script run.sh is provided. The script will generate the different paradigm, run the HISI algorithm on the whole database and train the different deep learning conditions (may take several couple of days). The MNIST database is also provided in a zip file that the user has to un-compress before running the script. 

To launch the script, go to your file and run ./run.sh in your terminal. At the end of the simulation, all results will be saved into numpy arrays into the results folder. Use your favorite tool to plot the results. 
 
Note that the HISI has been developed using python3 on a mac, while the deep learning algorithm have been launched on a Linux desktop computer using python 2 and TensorFlow to benefit from the graphic card. Therefore run.sh take care of the two versions of python but it may produce different results as seen in our lab between the mac and the Linux system. The script also uses multi-threading of the CPU, if not sure, try running the code for a single thread by un-comment the code. 
