FROM <MISSING>

#Install gcc because we will need it to build some of the
#python dependencies
RUN apt-get update && apt-get install -y gcc
#Update pip (pythons package manager)
RUN python -m pip install --upgrade pip && pip install virtualenv

#Copy in our model
COPY irismdl /irismdl

#Install the model requirements
RUN pip install -r  <MISSING>

#Tell it how to run the model
CMD ["mlflow", "models", "serve", "-h", "0.0.0.0", "-m", "irismdl", "--env-manager=local"]