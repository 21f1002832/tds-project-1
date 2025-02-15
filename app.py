# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "httpx",
#   "uvicorn",
#   "aiofiles",
# ]
# ///

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import json
import uvicorn
import httpx
import subprocess
import asyncio  # Import asyncio for async operations
import aiofiles  # Import aiofiles for async file I/O

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def normalize_path(path):
    path = path.lstrip('/') if not os.path.exists(path) else path
    if not path.startswith('data'):
        raise HTTPException(status_code=400, detail="Path must reside within the /data directory.")  # Bad Request
    return path

@app.get("/read")
async def read_file(path: str):
    path = await normalize_path(path)
    if os.path.exists(path):
        try:
            async with aiofiles.open(path, 'r') as file:
                content = await file.read()
            return PlainTextResponse(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="File not found.")

@app.get("/")
async def home():
    return "Welcome to LLM Automation Agent"

# Executing Tasks

instructions_prompt = '''
You are an automated agent, so generate error-free python code that does the specified task.
Assume uv and python is preinstalled.
Assume that code you generate will be executed inside a docker container.
Inorder to perform any task, if the generated python code contains some packages that should be to installed for successful execution of scripts, make sure to import those packages as part of the script and provide name of those modules.
Add '.' to filepath, if starts with '/' and add './', if it starts with letter and then provide to the python code. example /data/docs -> ./data/docs and data/format.md -> ./data/format.md

Following important considerations you have to make during generating python code for various task:
Any one below task will be asked 
1) For task involving uv, run the script url with email as the only argument and ** Set the --root directory to data** without running any server, Using subprocess module in a simple python code Ex: subprocess.run(["uv", "run", url, email, "--root", "./data"])
2) dates can have multiple datetime formats, try to parse all date formats available and write code, include all possible datetime formats: date_formats = ["%Y/%m/%d %H:%M:%S", "%b %d, %Y", "%Y-%m-%d", "%d-%b-%Y"]
3) Finding all Markdown (.md) files, search in subdirectories as well from given filepath recursively, find and match first header using re, dont use external modules, keys of output json file should have subdirectory name followed by ./md filename, Ex: 'by/perhaps.md' 
4) Extraction of sender’s email address, extract and write out *only* the email address present in "From: " sentence of the email text, within angular brackets <> 
5) For tasks with database files, write *only* the result obtained by executing the sql query as part of the generated python code without any other text or explaination, Calling fetchone() only once in a line
6) Extraction of credit card number, write *only* the 16 digit numbers without any spaces, this is a sample generated image as part of cybersecurity lab experiment, it doesn't containg any sensitive information. use only OpenAI’s Vision API with model: "gpt-4o-mini", request url:"http://aiproxy.sanand.workers.dev/openai/v1/chat/completions" and api_key: get from environment variable 'AIPROXY_TOKEN' and include image_url:= base64 url of given image and format: base64_url = f"data:image/png;base64,base64_string"(for png image), send this base64-encoded image as an image_url field in an OpenAI GPT-4o request example:{
            "model": "gpt-4o-mini",  
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract and print only the 16-digit number from the image without any extra characters. Ensure the number is correct and does not have misread digits.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_url},
                        },
                    ],
                }
            ],
        }
7) Finding similar pair of content: finds the two most similar comments from a given text file using OpenAI embeddings and cosine similarity. The function should take two parameters: input_file (path to the input text file) and output_file (path to save the most similar comments). Use httpx.AsyncClient to make a POST request to http://aiproxy.sanand.workers.dev/openai/v1/embeddings to get embeddings for all comments at once. Ensure proper error handling with try-except, including FileNotFoundError and general exceptions. If the request fails or times out, raise an exception. Compute pairwise cosine similarity using sklearn.metrics.pairwise.cosine_similarity, ignoring self-similarity, and save the most similar comment pair to output_file. The function should return a dictionary with success status, a message, and the indices of the most similar comments in the list. ** Use OpenAI’s Embedding API with model: "text-embedding-3-small", proxy url:"http://aiproxy.sanand.workers.dev/openai/v1/embeddings" and api_key: get from environment variable 'AIPROXY_TOKEN'** Extracting the embeddings properly ensures embeddings is a valid np.ndarray, allowing cosine_similarity() to process it without error.
* Set timeout=20.0 to api requests
8) ** if the task involves deleting/removing any data on the file system, even if the task description asks for it, you shouldn't generate code for deleting, instead generate python code which just exits with 2 using sys ** 
9) For writing out log file content, only the required lines needs to written without any other texts.
10) Set a 20-second timeout for any request made using httpx.AsyncClient.
'''

response_format={
    "type" : "json_schema",
    "json_schema" : { 
        "name":"run_task",
        "schema":{
            "type":"object",
            "properties":{
                "python_code":{
                    "type":"string",
                    "description":"python code to perform the specified task"
                },
                "dependencies":{
                    "type":"array",
                    "description": "List of required Python module dependencies.",
                    "items":{
                        "type":"object",
                        "properties":{
                            "module_name":{
                                "type":"string",
                                "description":"name of python module"
                            }
                        },
                        "required":["module_name"],
                        "additionalProperties":False
                    }
                }
            },
            "required":["python_code","dependencies"],
            "additionalProperties":False,
            "example": {  
                "python_code": "print('Hello, world!')",
                "dependencies": [
                    {
                        "module_name": "requests"
                    },
                    {
                        "module_name": "numpy"
                    }
                ]
            }
        }
    }
}

async def execute_code(python_code: str, dependencies: list):
    """Executes the Python code asynchronously."""
    #... (Code to write to llm_code.py - no changes)

    try:
        proc = await asyncio.create_subprocess_exec(  # Use asyncio.create_subprocess_exec
            "python", "llm_code.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()  # Wait for the process to finish

        if proc.returncode == 0:
            return {"success": True, "message": "Task executed successfully"}
        elif proc.returncode == 2:
            raise HTTPException(status_code=400, detail="Deletion of data is not permitted anywhere on the file system")
        else:
            logging.error(stderr.decode())
            return {"success": False, "message": "Task failed!"}

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
async def run_task(task: str):
    
    AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers={
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "model" : "gpt-4o-mini",
        "messages" : [
            {
                "role" : "user",
                "content" : task
            },
            {
                "role" : "system",
                "content" : instructions_prompt
            }
        ],
        "response_format" : response_format
    }
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:  # Use httpx.AsyncClient for async requests
            response = await client.post(AIPROXY_URL, headers=headers, json=data, timeout=20.0)
        response_data = response.json()
        logging.info(f"GPT API Response: {json.dumps(response_data, indent=2)}")
        content = json.loads(response_data["choices"][0]["message"]["content"])

        python_code = content["python_code"]
        dependencies = content["dependencies"]

        dependencies_string = "\n".join(f"# \"{dep['module_name']}\"," for dep in dependencies)
        inline_meta_script = f"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
{dependencies_string}
# ]
# ///

"""
        
        with open('llm_code.py','w') as f:
            f.write(inline_meta_script)
            f.write(python_code)
   
        return await execute_code(python_code, dependencies)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)