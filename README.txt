I. NEEDED TOOLS:
    a. Python 2.7 (3.* is not supported yet)
    b. numpy http://www.scipy.org/install.html | matrix related, kind a Matlab mapping for Python
    c. scipy http://www.scipy.org/install.html | needed by numpy
    d. matplotlib http://matplotlib.org | for plotting

    P.S: You can just download the latest IPython package, it contain all needed things. Check http://ipython.org/install.html
    P.S2: If you dont have matplotlib installed, program will get error message in plotting stage, but you still can see all TEXT output results.

II. HOW TO RUN:
    There are three executable python scripts, which are kNearestNeighbor.py , LogisticRegression.py and Plot.py

    1. Before run
        a. setup the needed tools
        b. start a terminal (or cmd in windows), make sure you are in the code folder [project2]. If you are using IPython, just open an IPython console, direct to the code folder.
        c. (OPTIONAL) put NEEDED database file into the [datasets] folder. If you dont do this step, program will download needed database automatically.

    2. Execute
        a. for k-NN:
        -----------------------------------------
        python kNearestNeighbor.py
        -----------------------------------------

        OR if you are using IPython console:
        -----------------------------------------
        run kNearestNeighbor.py
        -----------------------------------------

        b. for Logistic Regression:
        -----------------------------------------
        python LogisticRegression.py
        -----------------------------------------

        OR if you are using IPython console:
        -----------------------------------------
        run LogisticRegression.py
        -----------------------------------------

        c. for Plotting the results
        -----------------------------------------
        python Plot.py
        -----------------------------------------

        OR if you are using IPython console:
        -----------------------------------------
        run Plot.py
        -----------------------------------------

III. FILES:
    1. dataloader.py: mat data loader, read matlab file into python
    2. kNearestNeighbor.py: k-NN with main method to test
    3. LogisticRegression.py: Logistic Regression with main method to test
    4. PrincipalComponentAnalysis.py: implement of PCA approach
    5. Plot.py: plotting
    6. __init__.py: just ignore it please
    7. project2\*.sav: data collected for plotting
    8. datasets\*.mat: NEEDED DATABASES

