# set-up the virtual environment for this project
virtualenv venv
source venv/bin/activate
pip install -U pip
deactivate
source venv/bin/activate
pip3 install -r requirements.txt
