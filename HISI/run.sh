#!/bin/bash
echo "------------------------------------------------"
echo "Running the simulation for the project HISI (Human inspired segmentation and interpolation)"
echo "The purpose of the script is to reproduce all the deep learning results of the paper"
echo "The script will create all database and run all simulations, it may take a while (days)"
echo "Please refer to internal codes and scripts for details"
echo "------------------------------------------------"
echo

START=$(date +%s.%N)


# ****************************************************** #
# 1: Create the paradigms (threshold database and add 3 bar to train and 4 bar to test)
# ****************************************************** #

if [[ ! (-f "Stimuli/mnist_train.npy" && -f "Stimuli/mnist_test.npy" && -f "Stimuli/mnist_train_3bar.npy" && -f "Stimuli/mnist_test_4bar.npy") ]]; then
  	echo "Missing paradigm files, proceed to generate the paradigms"
	echo
	python3 mnist_bar.py
	echo "Paradigms generated"
	echo
else
	echo "files paradigms already exists! Nothing to generate"
	echo
fi


# ****************************************************** #
# 2: Run HISI on the MNIST paradigms
# ****************************************************** #

if [[ ! (-f "Stimuli/mnist_train_HISI.npy" && -f "Stimuli/mnist_train_3bar_HISI.npy" && -f "Stimuli/mnist_test_HISI.npy" && -f "Stimuli/mnist_test_4bar_HISI.npy") ]]; then
  	echo "Missing HISI files, proceed to their generation"
	echo

    # mnist_train_HISI
	mkdir temp
	python3 run_mnist_HISI.py mnist_train False False False True False
	python3 merge_results.py mnist_train_HISI False
	rm -rf temp

    # mnist_train_3bar_HISI
	mkdir temp
	python3 run_mnist_HISI.py mnist_train_3bar False True False True False
	python3 merge_results.py mnist_train_3bar_HISI False
	rm -rf temp

    # mnist_test_HISI
	mkdir temp
	python3 run_mnist_HISI.py mnist_test False False True True False
	python3 merge_results.py mnist_test_HISI True
	rm -rf temp

	# mnist_test_HISI
	mkdir temp
	python3 run_mnist_HISI.py mnist_test False False True True False
	python3 merge_results.py mnist_test_HISI True
	rm -rf temp

	# mnist_test_4bar_HISI
	mkdir temp
	python3 run_mnist_HISI.py mnist_test_4bar False True True True False
	python3 merge_results.py mnist_test_4bar_HISI True
	rm -rf temp


	echo "HISI paradigms generated"
	echo
else
	echo "HISI files already exist! Nothing to generate"
	echo
fi


# ****************************************************** #
# 3: Create double paradigms
# ****************************************************** #

if [[ ! ( -f "Stimuli/mnist_double_train_train.npy" && -f "Stimuli/mnist_double_train_3bar_HISI.npy" && -f "Stimuli/mnist_double_train_3bar_train_3bar.npy" && -f "Stimuli/mnist_double_train_3bar_train.npy" &&  -f "Stimuli/mnist_double_test_test.npy" && -f "Stimuli/mnist_double_test_4bar_HISI.npy" && -f "Stimuli/mnist_double_test_4bar_test_4bar.npy" && -f "Stimuli/mnist_double_test_4bar_test.npy") ]]; then
    echo "Missing double conditions, proceed to their generation"
	echo
	python construct_double_conditions.py
	echo "double paradigms generated"
	echo
else
	echo "double paradigms files already exists! Nothing to do"
	echo
fi


# ****************************************************** #
# 4: Create the random batch for CNN
# ****************************************************** #

if [[ ! -f "Stimuli/batch.npy" ]]; then
  	echo "Missing batch file, proceed to generate it"
	echo
	python mnist_batch.py
	echo "batch generated"
	echo
else
	echo "batch file already exists! Nothing to do"
	echo
fi


# ****************************************************** #
# 5: Run CNNs
# ****************************************************** #\

#simple conditions
if [[ ! -f "results/accuracy_mnist_train_mnist_test.npy" ]]; then
    echo "do: mnist_train mnist_test"
    python CNN.py mnist_train mnist_test
fi

if [[ ! -f "results/accuracy_mnist_train_mnist_test_4bar.npy" ]]; then
    echo "do: mnist_train mnist_test_4bar"
    python CNN.py mnist_train mnist_test_4bar
fi

if [[ ! -f "results/accuracy_mnist_train_3bar_mnist_test.npy" ]]; then
    echo "do: mnist_train_3bar mnist_test"
    python CNN.py mnist_train_3bar mnist_test
fi

if [[ ! -f "results/accuracy_mnist_train_3bar_mnist_test_4bar.npy" ]]; then
    echo "do: mnist_train_3bar mnist_test_4bar"
    python CNN.py mnist_train_3bar mnist_test_4bar
fi

if [[ ! -f "results/accuracy_mnist_train_mnist_test_4bar_HISI.npy" ]]; then
    echo "do: mnist_train mnist_test_4bar_HISI"
    python CNN.py mnist_train mnist_test_4bar_HISI
fi

if [[ ! -f "results/accuracy_mnist_train_3bar_HISI_mnist_test.npy" ]]; then
    echo "do: mnist_train_3bar_HISI mnist_test"
    python CNN.py mnist_train_3bar_HISI mnist_test
fi

if [[ ! -f "results/accuracy_mnist_train_3bar_HISI_mnist_test_4bar_HISI.npy" ]]; then
    echo "do: mnist_train_3bar_HISI mnist_test_4bar_HISI"
    python CNN.py mnist_train_3bar_HISI mnist_test_4bar_HISI
fi

#double conditions
if [[ ! -f "results/accuracy_mnist_double_train_3bar_train_mnist_double_test_4bar_test.npy" ]]; then
    echo "do: mnist_double_train_3bar_train mnist_double_test_4bar_test"
    python CNN_double.py mnist_double_train_3bar_train mnist_double_test_4bar_test
if

if [[ ! -f "results/accuracy_mnist_double_train_3bar_train_mnist_double_test_4bar_test_4bar.npy" ]]; then
    echo "do: mnist_double_train_3bar_train mnist_double_test_4bar_test_4bar"
    python CNN_double.py mnist_double_train_3bar_train mnist_double_test_4bar_test_4bar
fi

if [[ ! -f "results/accuracy_mnist_double_train_3bar_train_3bar_mnist_double_test_4bar_test.npy" ]]; then
    echo "do: mnist_double_train_3bar_train_3bar mnist_double_test_4bar_test"
    python CNN_double.py mnist_double_train_3bar_train_3bar mnist_double_test_4bar_test
fi

if [[ ! -f "results/accuracy_mnist_double_train_3bar_train_3bar_mnist_double_test_4bar_test_4bar.npy" ]]; then
    echo "do: mnist_double_train_3bar_train_3bar mnist_double_test_4bar_test_4bar"
    python CNN_double.py mnist_double_train_3bar_train_3bar mnist_double_test_4bar_test_4bar
fi

if [[ ! -f "results/accuracy_mnist_double_train_3bar_train_mnist_double_test_4bar_HISI.npy" ]]; then
    echo "do: mnist_double_train_3bar_train mnist_double_test_4bar_HISI"
    python CNN_double.py mnist_double_train_3bar_train mnist_double_test_4bar_HISI
fi

if [[ ! -f "results/accuracy_mnist_double_train_3bar_HISI_mnist_double_test_4bar_test.npy" ]]; then
    echo "do: mnist_double_train_3bar_HISI mnist_double_test_4bar_test"
    python CNN_double.py mnist_double_train_3bar_HISI mnist_double_test_4bar_test
fi

if [[ ! -f "results/accuracy_mnist_double_train_3bar_HISI_mnist_double_test_4bar_HISI.npy" ]]; then
    echo "do: mnist_double_train_3bar_HISI mnist_double_test_4bar_HISI"
    python CNN_double.py mnist_double_train_3bar_HISI mnist_double_test_4bar_HISI
if



END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

echo "Simulation finished! Simulation time:" $DIFF


