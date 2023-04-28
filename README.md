# llm-fastapi-example

Create API endpoints for question answering and sentiment analysis.

<p align="center">
<img src="images/llama.jpg" width=600>
<p/>

Photo: Shutterstock

## Table of Contents
1. [File Descriptions](#files)
2. [Supporting Packages](#packages)
3. [How To Use This Repository](#howto)
4. [Project Motivation](#motivation)
5. [About The Dataset](#data)
6. [Acknowledgements](#acknowledgements)
7. [Licence & copyright](#license)

## File Descriptions <a name="files"></a>
| File | Description |
| :--- | :--- |
| data/coltrane.txt | text from the John Coltrane Wikipedia page |
| data/index.json | sentence embeddings of text from coltrane.txt |
| main.py | FastAPI app |
| models.py | Hugging Face models and supporting functions |
| requirements.in | pip-tools spec file for requirements.txt |
| requirements.txt | list of python dependencies |

## Supporting Packages <a name="packages"></a>
In addition to the standard python library, this analysis utilizes the following packages:
- [FastAPI](https://fastapi.tiangolo.com/)
- [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/index.html)
- [safetensors](https://github.com/huggingface/safetensors)
- [SentenceTransformers](https://www.sbert.net/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://pypi.org/project/transformers/)
- [Uvicorn](https://www.uvicorn.org/)

Please see `requirements.txt` for a complete list of packages and dependencies used in the making of this project.

## How To Use This Repository <a name="howto"></a>
1. Clone the repo locally or download and unzip this repository to a local machine.
2. Navigate to this directory and open the command line. For the purposes of running the scripts, this will be the root directory.
3. Create a virtual environment to store the supporting packages.

        python -m venv venv --upgrade-deps

4. Activate the virtual environment.

        venv\scripts\activate

5. Install the supporting packages from the requirements.txt file.

        pip install -r requirements.txt
        
6. Run "uvicorn main:app --reload"

7. Open up a browser and go to 127.0.0.1:8000/docs and try out the two endpoints.
       
Note: if you do not have the hugging face models already, they will be downloaded and cached on your machine in the `C:\Users\USEDID\.cache` folder

## Project Motivation <a name="motivation"></a>
It is interesting to consider indexing a corpus with sentence embeddings and using them to query an LLM in a closed loop way, without calls to an external web API. This repo contributes an example in how to do this for the tasks of question answering from a knowledge base, as well as sentiment analysis.

## About The Dataset <a name="data"></a>
 See [John Coltrane Wikipedia page](https://en.wikipedia.org/wiki/John_Coltrane).

## Acknowledgements <a name="acknowledgements"></a>
Thanks to the developers who created the foundations for this example.

## License & copyright <a name="license"></a>
© Zachary Wolinsky 2023

Licensed under the [MIT License](LICENSE.txt)
