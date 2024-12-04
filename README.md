GIT IGNORE
Remember to add .csv files to git.ignore file:
  1. Find git.ignore file in your project folder
  2. Open it and add *.csv to the end (this will exclude all .csv files from being commited to repository)

INSTALL REQUIREMENTS FROM REQUIREMENTS.TXT
It is recommended to create virtual enviroment (.venv).
To install all python libraries at once use next command:
  pip install -r /path/to/requirements.txt


Folders description:

    web-site/ - everything related to website, done by Mykhailo
    
    old_files/ - storage for files that are not used in the final solution
        Vladimir/ - everything that Vladimir submitted
        tests.ipynb - some data EDA, made by Kirill and Hezekiah
        models_test_versions/ - stores models and working notebooks, made by Kirill and Hezekiah
        marwis_mobile_join.ipynb - data prep for marwis dataset, were not used in the end
        
    final/ - stores final solution, made by Kirill
        model_train.ipynb - training notebook
        demo.py - demo GUI
        data_prep.ipynb - data prep for starwis and weather datasets
        
    ready_data/ - recommended path to use for storing processed data
    original_data/  - recommended path to store original data
