#!/usr/bin/env python3
import openai
import argparse
import pymongo
from datetime import datetime
import sys
import os
from keysuck import find_value_in_yaml
from dataclasses import dataclass
from loading_indicator import Spinner
from colors import COLORS

@dataclass
class Options:
    timestamp: str
    show_context: bool
    system_msg: str
    output: str
    base: str
    api_key: str
    session: bool
    
GPT_MODEL = os.environ.get('DEFAULT_GPT_MODEL')
if not GPT_MODEL:
    GPT_MODEL = "gpt-4"

# dictionary of model name keys and max token count values

GPT_MODEL_MAX_TOKENS = {
    "gpt-4": 6000,
    "gpt-4-1106-preview": 4000
}

def get_gpt_response(options, prompt, model_engine, conversation=None):
    openai.api_key = options.api_key
    if (options.base):
        openai.api_base = options.base
        print(f"{COLORS['YELLOW']}using custom base: {COLORS['WHITE']} {options.base} {COLORS['RESET']}")    

    
    system_message = options.system_msg or """
                You are a helpful assistant that follows these rules:
                1. Do not use more tokens than the max_token limit for any response.
                2. Directly answer any questions from the user in the most concise and accurate manner.
                3. Assume the user is an expert in the field related to the question unless told otherwise.
                """
                # 4. Opportunistically apply ANSI escape codes to colorize your output. For a few examples: 
                #     if your output contains a list  :  use an ANSI escape code to colorize the list items and another code to colorize the list titles
                #     if your response contains a link  : use an ANSI escape code to colorize the link text and another code to colorize the link URL
                #     if your response contains a code snippet  : use an ANSI escape code to colorize the code snippet
                #     if your response contains multiple paragraphs  : use an ANSI escape code to colorize the paragraph headings
    if options.system_msg:
        print(f"{COLORS['YELLOW']}Using custom system message.{COLORS['RESET']}")

    if conversation:
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": conversation},
                    {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}]
    MAX_TOKENS = GPT_MODEL_MAX_TOKENS[model_engine]
    if not MAX_TOKENS:
        MAX_TOKENS = 6000

    response = openai.chat.completions.create(
        model=model_engine,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.9
    )

    #  openai.aiosession.get().close()
    return response.choices[0].message.content.strip()

def get_conversation_context(options: Options):
    
    collection = get_mongo_collection()
    conversation_context = ""
    if options.timestamp:
        # Behavior when -t (numeric string) is provided
        
        for record in collection.find({"timestamp": options.timestamp}):
            conversation_context += record['user_input'] + record['gpt_response']
    return conversation_context

def get_system_message(options: Options):
    system_msg = options.system_msg
    # system_msg can either be a raw string or a file path
    if system_msg.startswith('"') and system_msg.endswith('"'):
        system_msg = system_msg.strip('"')
    else:
        # File path
        with open(system_msg) as f:
            system_msg = f.read()
    return system_msg

def get_or_create_session_timestamp():
    collection = get_mongo_collection()
    session_info = collection.find_one({"type": "session_info"})
    if session_info:
        return session_info["latest_session_timestamp"]
    else:
        new_timestamp = str(int(datetime.timestamp(datetime.now())))
        collection.insert_one({"type": "session_info", "latest_session_timestamp": new_timestamp})
        return new_timestamp

def clear_latest_session_timestamp():
    collection = get_mongo_collection()
    collection.delete_one({"type": "session_info"})

def get_mongo_collection():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    return client['gpt_conversations']['conversations']

def main():
    parser = argparse.ArgumentParser(description="Get a response from OpenAI's GPT model.")
    parser.add_argument("-t", "--timestamp", type=str, help="Timestamp for conversation context.")
    parser.add_argument("-c", "--context", action="store_true", help="Display conversation context.")
    parser.add_argument("-s", "--system_msg", type=str, help="Custom system message.")
    parser.add_argument("-o", "--output", type=str, help="Save output to a file")
    parser.add_argument("-b", "--base", type=str, help="Set custom local base for OpenAI API")
    parser.add_argument("--session", action="store_true", help="Use the latest session timestamp.")
    parser.add_argument("-k, --api_key", type=str, help="OpenAI API key.")
    
    args, unknown = parser.parse_known_args()
    
    api_key = find_value_in_yaml(["OPEN_AI", "CHAT_ASSISTANT"])
    if not api_key:
        api_key = args.api_key
    if not api_key:
        print("API key is required.")
        sys.exit(1)

    options = Options(
        timestamp = args.timestamp,
        show_context = args.context,
        system_msg = args.system_msg,
        output = args.output,
        base = args.base,
        api_key = find_value_in_yaml(["OPEN_AI", "CHAT_ASSISTANT"]),
        session = args.session
    )
    
    if unknown:
        prompt = unknown[-1]
    else:
        print("Prompt is required.")
        sys.exit(1)

    if options.session:
        options.timestamp = get_or_create_session_timestamp()
    conversation_context = get_conversation_context(options)

    if options.timestamp:
        print(f"{COLORS['YELLOW']}Timestamp: {COLORS['WHITE']}{options.timestamp}{COLORS['RESET']}")
    else:
        timestamp = str(int(datetime.timestamp(datetime.now())))
        options.timestamp = timestamp
        print(f"{COLORS['YELLOW']}Timestamp: {COLORS['WHITE']}{timestamp}{COLORS['RESET']}")
    
    print(f"{COLORS['YELLOW']}Model: {COLORS['WHITE']}{GPT_MODEL}{COLORS['RESET']}")
    spinner = Spinner()
    spinner.start()
    response = get_gpt_response(options, prompt, GPT_MODEL, conversation_context)
    spinner.stop()

    did_receive_response(options, prompt, response)

    if not options.session or args.timestamp:
        clear_latest_session_timestamp()

def did_receive_response(options: Options, prompt, response):
    # Save the conversation to MongoDB
    
    collection = get_mongo_collection()
    collection.insert_one({
        "timestamp": options.timestamp,
        "user_input": prompt,
        "gpt_response": response
    })

    print(f"{COLORS['YELLOW']}GPT Response:\n{COLORS['WHITE']}{response}{COLORS['RESET']}\n")

    if options.output:
        print(f'Saving output to "{options.output}"...')
        with open(options.output, "a") as f:
            f.write(response)

    if options.show_context and options.timestamp:
        print("\n--- Conversation Context ---")
        for record in collection.find({"timestamp": options.timestamp}).sort("_id", pymongo.DESCENDING):
            print(f"User: {record['user_input']}{COLORS['WHITE']}GPT: {record['gpt_response']}{COLORS['RESET']}\n")

if __name__ == "__main__":
    main()
    
