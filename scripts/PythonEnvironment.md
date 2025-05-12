# Set up Python virtual environment for the first time
## Create the virtual environment
`python -m venv project_libs`

## Activate
`project_libs\Scripts\activate`

## Deactivate python environment
deactivate

# saving and install project dependencies
To save your installed packages to a requirements.txt file:
`pip freeze > requirements.txt`

To install from the requirements.txt file
`pip install -r requirements.txt`