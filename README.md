If you are not using Colab, the suggested method is to run this in a virtual environment:

# start your virtual environment
virtualenv --python=python3.10.2 env

# enter virtualenv
source env/bin/activate

# install requirements
pip install -r requirements.txt

# do your stuff...

# leave virtualenv
deactivate