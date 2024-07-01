# Setting up to run scripts using the openai API with Python.

The script will be run in a virtual environment. Start by creating a virtual environment:

On a Mac:
`python3 -m venv nptour-env`

On Windows:
`python -m venv nptour-env`

<br>
After creating the virtual environment, you need to activate it:

On a Mac:
`source nptour-env/bin/activate`

On Windows:
`nptour-env\Scripts\activate`

<br>
Once the virtual environment is activated, the beginning of your terminal prompt should display (nptour-env).

<br>
Install the LangChain modules by running (in both a Mac and Windows):

`pip install langchain langchain-core langchain-community langchain-openai lark langchain-pinecone pinecone-client langchainhub fuzzywuzzy python-Levenshtein python-dotenv lxml wikipedia flask` 

You'll see an openai-env folder has been added to the directory with all of the installed dependencies.

<br>

To run your code, in the command line run:

Without a debugger:
`flask run`

With a debugger:
`flask run --debug`

<br>
To stop the run, click control + C.
Then hard refresh the page. When making changes to your Python, HTML, or JavaScript code (and not using debugger) you'll need to stop the run after each change.

<br>
When finished, quit the run by clicking control + C and close the virtual environment by running: 

`deactivate`