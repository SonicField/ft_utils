# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-ignore-all-errors

import argparse
import asyncio
import atexit
import concurrent.futures
import contextlib
import curses
import datetime
import functools
import inspect
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import select
import signal
import sqlite3
import subprocess
import sys
import tempfile
import termios
import textwrap
import threading
import time
import traceback
import tty

from collections import deque, OrderedDict
from dataclasses import dataclass, make_dataclass
from random import randint
from pygments import highlight, token as lx_token
from pygments.formatters import Terminal256Formatter, TerminalFormatter
from pygments.lexer import RegexLexer
from pygments.lexers import get_lexer_by_name as lx_get_lexer_by_name, MarkdownLexer
from pygments.styles import get_all_styles


DEFAULT_MODEL = "llama3.3-70b-instruct"
PASTE_CLIENT_NAME = "scrolly"


KEY_ESCAPE = 27


# For short running, none blocking operations.
BACK_GROUND_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=multiprocessing.cpu_count()
)


# Use a pool size of one and this becomes a queue.
SUMMARY_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# Kill all pools on exit.
def kill_executor_pools():
    current_module = sys.modules[__name__]
    for member_name in dir(current_module):
        member = getattr(current_module, member_name)
        if isinstance(member, concurrent.futures.Executor):
            member.shutdown(wait=False)
    print("Waiting for threads to exit")
    time.sleep(1)
    print("Done waiting - Bye!")
    os._exit(0)


atexit.register(kill_executor_pools)


class Logger:
    MAX_ENTRIES = 1024

    def __init__(self):
        self.log = deque(maxlen=self.MAX_ENTRIES)
        self.lock = threading.Lock()

    def exception(self, e):
        with self.lock:
            entry = {
                "type": "Exception",
                "time": time.monotonic(),
                "text": repr(e),
                "stack_trace": self._format_stack_trace(e),
            }
            self.log.append(entry)

    def info(self, text):
        with self.lock:
            entry = {"type": "Information", "time": time.monotonic(), "text": text}
            self.log.append(entry)

    def warn(self, text):
        with self.lock:
            entry = {"type": "Warning", "time": time.monotonic(), "text": text}
            self.log.append(entry)

    def debug(self, text):
        if SCROLLY_OPTS.debug:
            with self.lock:
                entry = {"type": "Debug", "time": time.monotonic(), "text": text}
                self.log.append(entry)

    def get_log(self):
        with self.lock:
            return list(self.log)

    def _format_stack_trace(self, e):
        stack_trace = traceback.extract_tb(e.__traceback__)
        formatted_stack_trace = []
        for frame in stack_trace:
            formatted_frame = {
                "filename": frame[0],
                "lineno": frame[1],
                "name": frame[2],
                "line": frame[3],
            }
            formatted_stack_trace.append(formatted_frame)
        return formatted_stack_trace


LOGGER = Logger()


# Define constants for roles
class Role:
    AI = "AI"
    BASH = "BASH"
    COMMENT = "COMMENT"
    COOKED = "COOKED"
    FILE = "FILE"
    GENERATOR = "GENERATOR"
    SCL = "SCL"
    SYSTEM = "SYSTEM"
    TITLE = "TITLE"
    USER = "USER"

    @classmethod
    def is_valid(cls, role):
        return role in (
            cls.AI,
            cls.BASH,
            cls.COMMENT,
            cls.COOKED,
            cls.FILE,
            cls.GENERATOR,
            cls.SCL,
            cls.SYSTEM,
            cls.TITLE,
            cls.USER,
        )

    _colour_map = None

    @classmethod
    def _get_colour_map(cls):
        if cls._colour_map is None:
            cls._colour_map = {
                cls.AI: CursesDefs.BRIGHT_BLUE,
                cls.BASH: CursesDefs.BRIGHT_RED,
                cls.COMMENT: CursesDefs.BRIGHT_GREEN,
                cls.COOKED: CursesDefs.BRIGHT_LILAC | curses.A_ITALIC,
                cls.FILE: CursesDefs.BRIGHT_GOLD,
                cls.GENERATOR: CursesDefs.BRIGHT_MOSS,
                cls.SCL: CursesDefs.BRIGHT_AMETHYST,
                cls.SYSTEM: CursesDefs.BRIGHT_MAGENTA,
                cls.TITLE: CursesDefs.BRIGHT_CYAN,
                cls.USER: CursesDefs.BRIGHT_YELLOW,
            }
        return cls._colour_map

    @classmethod
    def get_role_colour(cls, role):
        return cls._get_colour_map()[role]


@dataclass
class DataFields:
    def __iter__(self):
        return iter(self.__dict__.values())

    def __getitem__(self, index):
        return list(self.__dict__.values())[index]

    def __len__(self):
        return len(self.__dict__)


def datafields(cls=None, *, bases=(DataFields,), defaults=None):
    def wrapper(cls):
        return make_dataclass(cls.__name__, cls.__fields__, bases=bases)

    if cls is None:
        return wrapper
    else:
        return wrapper(cls)


@datafields(defaults={"filename": None, "editor": None})
class PromptBase:
    __fields__ = "role", "text", "filename", "editor", "syntax", "summary"


class _Prompt(PromptBase):
    def copy(self):
        return Prompt(
            role=self.role,
            text=self.text,
            filename=self.filename,
            editor=self.editor,
            syntax=self.syntax,
            summary=self.summary,
        )


def Prompt(role, text, filename=None, editor=None, syntax=None, summary=None):
    return _Prompt(role, text, filename, editor, syntax, summary)


@datafields
class ConversationBase:
    __fields__ = "prompts", "scroll_offset"


class SystemPrompts:
    USE_MARKDOWN = [
        Prompt(
            role=Role.USER,
            text="""
# System Configuration Update
From this point forward, all interactions with this LLM will utilize Markdown formatting. This includes:
* User input prompts
* LLM responses
* Instruction sets

When processing user prompts, the LLM should:
* Parse Markdown syntax to extract semantic meaning
* Respect and utilize hyperlinks, emphasis, and other high-value elements
* Interpret headers, lists, and other formatting to inform response context

## Operation Mode: Markdown-Enhanced Semantic Processing
Enable enhanced semantic analysis to integrate Markdown elements into the meaning extraction process. This allows for more nuanced understanding and generation of text, incorporating the structural and contextual information provided by Markdown formatting.

## Confirm Understanding
Respond with a confirmation message in Markdown format, acknowledging the system configuration update and operation mode shift. Include a brief description of how you will utilize Markdown elements in your responses moving forward.
""",
        ),
        Prompt(
            role=Role.AI,
            text="""
# Confirmation of System Configuration Update
The system configuration update has been acknowledged, and the operation mode has been shifted to **Markdown-Enhanced Semantic Processing**.
* The LLM will parse Markdown syntax to extract semantic meaning from user input prompts.
* Hyperlinks, emphasis, and other high-value elements will be respected and utilized to inform response context.
* Headers, lists, and other formatting will be interpreted to generate more accurate and context-specific responses.
## Operation Mode Details
Moving forward, responses will be generated with consideration of the structural and contextual information provided by Markdown formatting, enabling more nuanced understanding and generation of text.
* **Hyperlinks** will be used to provide additional context or reference external information.
* **Emphasis** will be applied using Markdown syntax (e.g., *italic*, **bold**) to highlight important information or distinguish between different concepts.
* **Headers** will be utilized to organize and provide a clear structure for responses, making it easier for users to follow and understand the content.
* **Lists** will be used to present information in a clear and concise manner, facilitating easier reading and comprehension.
""",
        ),
    ]

    CREATE_SUMMARY = [
        Prompt(
            role=Role.USER,
            text="""
# Gaol
* I need a one line summary of the previous message.
* It needs to captue idenifiable information about the contenxts of the message.
* It can use markdown like bold and quote marks.

# Uses
* This will form part of a user interface.
* The user will see a list of these one line summaries for many messages.
* This will allow the user to choose between them without having to remember there content.

# Action
* Please write this summary for me now.
* Write one line of text
* Do not write titles, explanations or any other extraneous text
""",
        ),
    ]

    UNDERSTOOD = [
        Prompt(
            role=Role.AI,
            text="""**I shall use this knowledge to guide my future responses.**
* Please provide me with your next instructions.""",
        )
    ]

    SCL_SYNTAX = [
        Prompt(
            role=Role.USER,
            text="""
# Please read these instructions and act on them
* You may be asked to generate a programming language called SCL.
* You may be asked to write code in another language which generates SCL.
* If so, generate well-formed statements using the following formal grammar. Ensure that:

1. All commands end with : and are not whitespace-separated.
2. Tokens follow commands, separated by whitespace unless quoted.
3. Quoted tokens preserve whitespace and allow valid escape sequences.
4. Escape sequences are processed in tokens but NOT in commands.
5. Comments (#) are standalone statements.
6. Whitespace before a statement is ignored.
7. Statements should make sense logically and avoid unnecessary repetition.
---

Formal Grammar (EBNF)

program ::= (statement | comment | empty_line)*
statement ::= ws* command ":" (ws token)* ws* newline
comment ::= ws* "#" (ws token)* ws* newline
command ::= printable_unicode_except_whitespace+
                 | quoted_token (* Quotes are preserved in commands *)
token ::= quoted_token | unquoted_token | control_char_token
quoted_token ::= ("\"" qchar* "\"" ) | ("'" qchar* "'")
qchar ::= any_unicode_except_quote_or_backslash
                 | "\\" escape_sequence
unquoted_token ::= uchar+ (* unquoted tokens process escapes *)
uchar ::= any_unicode_except_whitespace_or_control_or_backslash
                 | "\\" escape_sequence
escape_sequence ::= "\\" | "n" | "r" | "v" | "t"
                 | "x" hex_digit hex_digit
                 | "u" hex_digit hex_digit hex_digit hex_digit
                 | "U" hex_digit hex_digit hex_digit hex_digit
                   hex_digit hex_digit hex_digit hex_digit
control_char_token ::= "\n" | "\r" | "\t" | "\v"
empty_line ::= ws* newline
ws ::= " " | "\t"
newline ::= "\n" | "\r\n"
hex_digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7"
                 | "8" | "9" | "A" | "B" | "C" | "D" | "E" | "F"
                 | "a" | "b" | "c" | "d" | "e" | "f"
---

Example Valid Outputs

✅ Basic Commands

start:
run: task1 task2 task3
execute: "A quoted token with spaces"

Commands end with :
Tokens follow commands, separated by whitespace unless quoted
Quoted tokens retain whitespace and allow escape sequences
---

✅ Using Escape Sequences Correctly

path: "C:\\Users\\Alice"
message: "Hello\nWorld!"
unicode_test: "\u2764 \U0001f600"
escaped: "This is a double backslash: \\\\"

Escape sequences are processed inside tokens
Backslashes (\\) are preserved where needed
---

✅ Control Characters as Tokens

split: token1 \n token2
indented: \t "Indented text"
vertical_tab: "This\vThat"

\n, \t, and \v can be standalone tokens
Inside quoted tokens, escape sequences are processed
---

✅ Comments and Empty Lines

# This is a comment
execute: task1 task2

# Another comment
run: task3 task4

Comments start with # and do not affect parsing
Empty lines are ignored unless inside a quoted token
---

✅ Handling Quoted Commands (Quotes are Preserved)

"weird_command:": arg1 arg2
'also_weird_command:': "Quoted argument"

Quoted commands are valid, and their quotes are preserved
They follow the same rules as unquoted commands
---

Important Rules to Follow

❌ Never generate a command without a : at the end.
❌ Do not put spaces inside an unquoted command.
❌ Escape sequences should only be processed inside tokens.
❌ Commands should not process escape sequences.
❌ Ensure statements make sense (avoid nonsense or redundant tokens).
---

Example Task and Response

Task Prompt:

"Generate a script that sets a file path, prints a message, and includes a comment."

Expected AI Output:

# This script sets a file path and prints a message
set_path: "/Users/Alice/Documents"
print: "Welcome!\nEnjoy your stay."
---

Final Instruction
* Generate well-formed statements that follow the grammar precisely.
* Ensure logical consistency in generated content.
* Never generate incorrect syntax or missing colons.
* Escape sequences should be properly handled inside tokens.
* Output should be clean, readable, and directly usable.
""",
        ),
        UNDERSTOOD[0],
        Prompt(
            role=Role.USER,
            text="""
# Index-Range Micro Language
Index-range micro language, which allows users to specify a range of indices using a concise syntax. The language is designed to parse a comma-delimited toke into a list of integer indices.

## Syntax
The index-range micro language supports the following syntax elements:

* **Single integer**: A single integer value, e.g., `1`.
* **Range**: A range of integers, specified using a dash (`-`) separator, e.g., `1-3`.
* **Open-ended range**: A range that starts or ends at an unspecified index, e.g., `1-` or `-3`.
* **Relative ranges**: Ranges that are relative to a current index, specified using a dot (`.`) or a dollar sign (`$`), e.g., `.-3` or `$-3`.
* **Relative offsets**: Offsets from a current index, specified using a plus sign (`+`) or a tilde (`~`), e.g., `+3` or `~3`.
* **Current index**: A special symbol (`.`) that represents the current index.
* **Last index**: A special symbol (`$`) that represents the last valid index.

## Formal Grammar
The formal grammar for the index-range micro language can be defined as follows:
```markdown
# Index-Range Micro Language Grammar

 Terminals
* `INT`: a non-negative integer value
* `DOT`: a dot (`.`) symbol
* `DOLLAR`: a dollar sign (`$`) symbol
* `PLUS`: a plus sign (`+`) symbol
* `TILDE`: a tilde (`~`) symbol
* `DASH`: a dash (`-`) symbol
* `COMMA`: a comma (`,`) symbol

 Non-Terminals
* `EXPR`: an expression
* `RANGE`: a range
* `OFFSET`: an offset
* `RELATIVE`: a relative expression

 Productions
1. `EXPR` ::= `INT` | `RANGE` | `RELATIVE`
2. `RANGE` ::= `INT` `DASH` `INT` | `INT` `DASH` | `DASH` `INT`
3. `RELATIVE` ::= `DOT` `OFFSET` | `DOLLAR` `OFFSET`
4. `OFFSET` ::= `PLUS` `INT` | `TILDE` `INT`
5. `EXPR_LIST` ::= `EXPR` (`COMMA` `EXPR`)*

 Notes
* The `EXPR_LIST` production represents a comma-delimited list of expressions.
* The `RELATIVE` production allows for relative expressions using the dot (`.`) or dollar sign (`$`) symbols.
* The `OFFSET` production allows for offsets using the plus sign (`+`) or tilde (`~`) symbols.
```
## Semantics
The semantics of the index-range micro language are defined by the Python code, which parses the input string and generates a list of integer indices. The code handles errors and edge cases, such as invalid input syntax, out-of-range indices, and empty segments.

## Example Use Cases
The index-range micro language can be used in various applications, such as:

* Specifying a range of pages to print or display
* Selecting a subset of data from a larger dataset
* Defining a range of indices for a data structure or array

For example, the input string `"1-3, 5, 7-"` might be parsed into the list of indices `[1, 2, 3, 5, 7, 8, 9, ...]`, depending on the maximum valid index.### Index-Range Micro Language Examples
Here are 10 examples that demonstrate various aspects of the index-range micro language:

1. **Simple range**: `1-3`
 * Parses to: `[1, 2, 3]`
2. **Open-ended range**: `5-`
 * Parses to: `[5, 6, 7, ..., max_count - 1]`
3. **Relative range**: `.-2`
 * Parses to: `[current - 2, current - 1, current]` (assuming `current` is set)
4. **Relative offset**: `+3`
 * Parses to: `[current + 3]` (assuming `current` is set)
5. **Last index**: `$`
 * Parses to: `[max_count - 1]`
6. **Single index**: `7`
 * Parses to: `[7]`
7. **Multiple ranges**: `1-3, 5, 7-`
 * Parses to: `[1, 2, 3, 5, 7, 8, 9, ..., max_count - 1]`
8. **Relative range with offset**: `~2-`
 * Parses to: `[current - 2, current - 1, current, current + 1, ..., max_count - 1]` (assuming `current` is set)
9. **Empty range**: `-`
 * Parses to: `[0, 1, 2, ..., max_count - 1]` (all indices)
10. **Reverse range**: `3-0`
 * Parses to: `[3, 2, 1, 0]`
11. **Complex example**: `1-3, 5, $-2, .+1`
 * Parses to: `[1, 2, 3, 5, max_count - 1, max_count - 2, current + 1]` (assuming `current` is set)

Notes:
* These examples assume a maximum valid index (`max_count`) and a current index (`current`) are set, unless otherwise specified. The actual parsing results may vary depending on the specific implementation and input values.
* `current` is set via the current: command in the SCL language. `current: 10` will set current to 10. Withoyt being set it defaults to 0.
* `max_count` in SCL is set automatically to the number of prompts in a conversation or text file.
""",
        ),
        UNDERSTOOD[0],
    ]

    COOKED = [
        Prompt(
            role=Role.SYSTEM,
            text="""For the following prompt please write a plan of how you would answer the prompt; do not actually answer the prompt.""",
        ),
    ]

    UNCOOKED = [
        Prompt(
            role=Role.USER,
            text="""Please read the plan you wrote and execute that plan. Do not mention the plan; write just the outcome.""",
        ),
    ]

    DEFAULT = []


class Conversation(ConversationBase):
    def copy(self):
        return Conversation(
            prompts=[prompt.copy() for prompt in self.prompts],
            scroll_offset=self.scroll_offset,
        )


class PersistentLRUCache:
    def __init__(self, db_path, maxsize):
        self.maxsize = maxsize
        self.db_path = db_path
        self.cache = OrderedDict()
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.commit()
        self.load_cache_from_db()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            key = self._get_key(args, kwargs)
            if key in self.cache:
                result = self.cache.pop(key)
                self.cache[key] = result
                return result
            else:
                result = func(*args, **kwargs)
                self.add(key, result)
                return result

        return wrapper

    def _get_key(self, args, kwargs):
        return str((args, frozenset(kwargs.items())))

    def add(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            self._evict_lru()
        self.cache[key] = value
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        else:
            self.cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
            result = self.cursor.fetchone()
            if result:
                value = result[0]
                self.add(key, value)
                return value
        return None

    def _evict_lru(self):
        lru_key = next(iter(self.cache))
        self.cache.pop(lru_key)
        self.cursor.execute("DELETE FROM cache WHERE key = ?", (lru_key,))
        self.conn.commit()

    def load_cache_from_db(self):
        self.cursor.execute("SELECT key, value FROM cache")
        results = self.cursor.fetchall()
        for key, value in results:
            self.cache[key] = value
        while len(self.cache) > self.maxsize:
            self._evict_lru()


SUMMARY_CACHE = None


def get_summary_cache():
    global SUMMARY_CACHE
    if SUMMARY_CACHE is None:
        SUMMARY_CACHE = PersistentLRUCache(
            f"{SCROLLY_OPTS.prompts_dir}/.summary_cache",
            SCROLLY_OPTS.summary_cache_size,
        )
    return SUMMARY_CACHE


class Yanked:
    def __init__(self):
        self._current_yank = None

    def get_current(self):
        if self._current_yank is not None:
            return self._current_yank.copy()
        else:
            return None

    def set_current(self, prompts):
        if prompts is not None:
            self._current_yank = prompts.copy()
            for prompt in self._current_yank:
                prompt.editor = None

        else:
            self._current_yank = None


def summarise_many(to_summarise):
    def set_summary():
        for prompt in (p for p in to_summarise if p.summary is None):
            try:
                background_summarise_prompt(prompt).result()
            except Exception as e:
                LOGGER.exception(e)

    try:
        BACK_GROUND_POOL.submit(set_summary)
    except Exception as e:
        LOGGER.exception(e)


class Conversations:
    def __init__(self, count):
        self._conversations = [
            Conversation(prompts=[], scroll_offset=0) for _ in range(count)
        ]
        self._current_index = 0
        self._deleted = []

    def _check_index(self, index):
        if index < 0 or index >= len(self._conversations):
            raise IndexError(f"Incorrect index for conversation slots {index}")

    def get_conversation(self, index):
        self._check_index(index)
        return self._conversations[index].copy()

    def set_conversation(self, index, conversation):
        self._check_index(index)
        self._conversations[index] = conversation.copy()
        summarise_many(self._conversations[index].prompts)

    def get_current_conversation(self):
        return self.get_conversation(self.get_current_index())

    def set_current_conversation(self, conversation):
        self.set_conversation(self.get_current_index(), conversation)

    def get_current_index(self):
        return self._current_index

    def set_current_index(self, index):
        self._current_index = index

    def add_deleted(self, prompt):
        self._deleted.append(prompt.copy())

    def undelete(self, index_range):
        index_range = sorted(set(parse_index_range(index_range, len(self._deleted))))
        ret = [self._deleted[idx].copy() for idx in index_range]
        summarise_many(ret)
        for idx in reversed(index_range):
            del self._deleted[idx]
        return ret

    def __iter__(self):
        return (conv.copy() for conv in self._conversations)

    def __getitem__(self, index: int):
        return self.get_conversation(index)

    def __len__(self):
        return len(self._conversations)


class Status:
    last_loaded = None
    last_read = None
    last_written = None
    last_saved = None
    conversations = Conversations(10)
    yanked = Yanked()


async def send_query(messages):
    # Initialize the MetaGen API
    metagen_api = thrift_platform_factory.create(
        metagen_auth_credential=MetaGenKey(key=SCROLLY_OPTS.key),
        auto_rate_limit=True,
    )
    # Generate a response from Metamate
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        metagen_api.chat_completion,
        messages,
        SCROLLY_OPTS.model,
        SCROLLY_OPTS.temperature,
        SCROLLY_OPTS.top_p,
        SCROLLY_OPTS.max_tokens,
    )
    return response.choices[0].text


async def update_query_screen(stdscr, start_time):
    while True:
        height, width = stdscr.getmaxyx()
        elapsed_time = time.time() - start_time
        text = f"Thinking {elapsed_time:.1f}"
        text_len = len(text)
        win_height = 5
        win_width = text_len + 4
        win_y = ((height - win_height) - win_height) // 2
        win_x = ((width - win_width) - win_width) // 2
        win = curses.newwin(win_height, win_width, win_y, win_x)
        win.clear()
        win.border()
        win.addstr(2, 2, text, curses.color_pair(CursesDefs.GREEN))
        win.refresh()
        await asyncio.sleep(0.1)


def extract_bash_script(text):
    # Using none greedy matching so we consume to the first closing pattern.
    pattern = r"^```bash\n([\s\S]*?)\n```$"
    matches = re.findall(pattern, text, re.MULTILINE)

    if len(matches) > 1:
        raise ValueError("Ambiguous bash information")
    elif len(matches) == 1:
        return matches[0]
    else:
        return text


async def handle_execute_async(stdscr, prompts):
    height, width = stdscr.getmaxyx()
    send_messages = []
    full_prompts = prompts[:-1]
    if prompts[-1].role == Role.GENERATOR:
        full_prompts.extend(SystemPrompts.SCL_SYNTAX)
        docs = SCLAutomator.document()
        commands = [
            Prompt(
                role=Role.USER,
                text=f"""Here is the definition of all the SCL commands.
Where ever an token is called `index_range` this token follows the index range syntax you just read about.

{create_table_string("SCL Help:", docs, False)}""",
            )
        ] + SystemPrompts.UNDERSTOOD
        full_prompts.extend(commands)
    elif prompts[-1].role == Role.COOKED:
        full_prompts.extend(SystemPrompts.COOKED)
    full_prompts.append(prompts[-1])

    for message in full_prompts:
        if message.role in (Role.USER, Role.FILE, Role.GENERATOR, Role.COOKED):
            role = ChatRole.USER
            text = message.text
        elif message.role == Role.SYSTEM:
            role = ChatRole.USER
            text = f"""**DO NOT RESPOND** - The following text is for context only and does not require a response:

 {message.text}"""
        elif message.role == Role.BASH:
            role = ChatRole.USER
            text = f"""**DO NOT RESPOND** - The following text is a bash script. Please read but do not respond. The prompt after it will be the output from the bash script:

 {message.text}"""
        elif message.role == Role.AI:
            role = ChatRole.AI
            text = message.text
        elif message.role in (Role.COMMENT, Role.TITLE, Role.SCL):
            role = None
            text = None
        else:
            raise ValueError(f"Untranslatable role {message.role}")

        if role is not None and text is not None:
            if message.filename is not None:
                send_messages.append(
                    Message(
                        role=ChatRole.USER,
                        text=f"**DO NOT RESPOND** - the following prompt is the content of file {message.filename}",
                    )
                )
            send_messages.append(Message(role=role, text=text))

    start_time = time.time()
    curses.start_color()
    curses.use_default_colors()
    update_task = asyncio.create_task(update_query_screen(stdscr, start_time))
    result = await send_query(send_messages)
    update_task.cancel()
    try:
        await update_task
    except asyncio.CancelledError:
        pass
    return result


MAX_SUMMARY_LINES = 1000
SUMMARY_DELAY_SECCONDS = 10
SUMMARY_TEMP = 0.5
SYMMARY_TOP_P = 0.5


def background_summarise_prompt(prompt):
    async def query_task(messages):
        metagen_api = thrift_platform_factory.create(
            metagen_auth_credential=MetaGenKey(key=SCROLLY_OPTS.key),
            auto_rate_limit=True,
        )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            metagen_api.chat_completion,
            messages,
            DEFAULT_MODEL,
            SUMMARY_TEMP,
            SYMMARY_TOP_P,
            SCROLLY_OPTS.max_tokens,
        )
        return response.choices[0].text

    async def async_task():
        try:
            text = prompt.text.split("\n")
            # Seems like human's summarise log docs by looking a the top and the bottom, so let's do that.
            if len(text) > MAX_SUMMARY_LINES:
                text = "\n".join(
                    text[: MAX_SUMMARY_LINES // 2]
                    + ["\n"]
                    + text[-1 * MAX_SUMMARY_LINES // 2 :]
                )
                LOGGER.info("Summary text truncated.")
            else:
                text = prompt.text
            if prompt.filename:
                text = f"The Following text is the contexts of the file of name {prompt.filename}:\n\n{text}"
            elif prompt.role == Role.BASH:
                text = f"The Following text a bash script the user will run: \n\n```bash\n{text}```"
            elif prompt.role == Role.USER:
                text = f"The Following is a prompt the user has written. The text is between >>> tokens: \n\n>>>\n{text}\n>>>"
            elif prompt.role == Role.AI:
                text = f"The Following is a response the AI has written. The text is between >>> tokens: \n\n>>>\n{text}\n>>>"

            send_messages = [
                Message(role=ChatRole.USER, text=text),
                Message(role=ChatRole.USER, text=SystemPrompts.CREATE_SUMMARY[0].text),
            ]
            # On occasion we get a long summary, try passing it through the system again and see if that helps.
            summary = await query_task(send_messages)
            summary = summary.strip()
            if len(summary.splitlines()) > 1:
                send_messages = [
                    Message(role=ChatRole.USER, text=summary),
                    Message(
                        role=ChatRole.USER, text=SystemPrompts.CREATE_SUMMARY[0].text
                    ),
                ]
                summary = await query_task(send_messages)
                summary = summary.strip()
            return summary
        except Exception as e:
            LOGGER.exception(e)

    def thread_task():
        persistent_cache = get_summary_cache()

        @persistent_cache
        def send_it(text, role):
            ret = None
            try:
                LOGGER.info(
                    f"Summarising: {role}:{id(prompt)} @ {len(text.splitlines())} lines."
                )
                # Always wait a little to give the chance for forground queries to be prioritied.
                time.sleep(SUMMARY_DELAY_SECCONDS)
                ret = asyncio.run(async_task())
                LOGGER.info(f"Summerized: {id(prompt)}")
            except Exception as e:
                LOGGER.exception(e)
            return ret

        if prompt.role == Role.TITLE:
            LOGGER.info("Setting title summary to title top line")
            prompt.summary = prompt.text.splitlines()[0]
        else:
            prompt.summary = send_it(prompt.text, prompt.role)

    return SUMMARY_POOL.submit(thread_task)


def get_paste_client():
    return PhabricatorPaste(
        PhabricatorAuthStrategyFactory.paste_bot(), PASTE_CLIENT_NAME
    )


def create_paste(content, title, language="txt"):
    paste = get_paste_client().create_phabricator_paste_object(
        PASTE_CREATION_CLIENT_ID, content, title, language
    )
    return paste.url


def load_paste(paste_number):
    paste = get_paste_client().get_by_number(str(paste_number))
    return paste["raw_content"]


def create_file_path(directory_path, filename):
    file_path = pathlib.Path(directory_path) / filename
    absolute_path = file_path.resolve()
    return absolute_path


def validate_directory(path):
    validated_path = pathlib.Path(path).expanduser()

    if not validated_path.exists():
        answer = input(
            f"The directory '{validated_path}' does not exist, would you like to create it? (y) "
        )
        if answer.lower() == "y" or answer.lower() == "yes":
            validated_path.mkdir()
            return validate_directory(path)
        else:
            raise FileNotFoundError(f"The directory '{validated_path}' does not exist.")

    if not validated_path.is_dir():
        raise NotADirectoryError(f"The path '{validated_path}' is not a directory.")

    return validated_path


class Border:
    TOP_LEFT = "\u2552"
    TOP_RIGHT = "\u2555"
    BOTTOM_LEFT = "\u2558"
    BOTTOM_RIGHT = "\u255b"
    VERTICAL = "\u2502"
    HORIZONTAL_DOUBLE = "\u2550"
    DOUBLE_TO_SINGLE_LEFT = "\u255e"
    DOUBLE_TO_SINGLE_RIGHT = "\u2561"
    CROSS = "\u256a"
    DOUBLE_TO_SINGLE_TOP = "\u2564"
    DOUBLE_TO_SINGLE_BOTTOM = "\u2567"


def safe_addstr(stdscr, height, y, x, txt, colour=0):
    if y < height - 1 and y >= 0:
        try:
            stdscr.addstr(y, x, txt, colour)
        except curses.error:
            raise ValueError(f"Curses error for {(y, x, txt, colour)}")
    elif y == height - 1:
        try:
            stdscr.addstr(y, x, txt, colour)
        except curses.error:
            pass


class CursesDefs:
    @classmethod
    def setup(cls):
        curses.initscr()
        curses.start_color()
        curses.use_default_colors()

        # Basic Colour Pairs
        cls.BLACK = hex_to_xterm("000000")
        cls.CYAN = hex_to_xterm("008080")
        cls.MAGENTA = hex_to_xterm("800080")
        cls.YELLOW = hex_to_xterm("808000")
        cls.WHITE = hex_to_xterm("808080")
        cls.GREEN = hex_to_xterm("00A000")
        cls.RED = hex_to_xterm("A00000")
        cls.BLUE = hex_to_xterm("0000A0")

        cls.MEDIUM_WHITE = hex_to_xterm("B0B0B0")

        cls.BRIGHT_BLACK = hex_to_xterm("404040")
        cls.BRIGHT_CYAN = hex_to_xterm("00FFFF")
        cls.BRIGHT_MAGENTA = hex_to_xterm("FF00FF")
        cls.BRIGHT_YELLOW = hex_to_xterm("FFFF00")
        cls.BRIGHT_WHITE = hex_to_xterm("FFFFFF")
        cls.BRIGHT_GREEN = hex_to_xterm("00FF00")
        cls.BRIGHT_RED = hex_to_xterm("FF0000")
        cls.BRIGHT_BLUE = hex_to_xterm("0000FF")
        cls.BRIGHT_GOLD = hex_to_xterm("FF8844")
        cls.BRIGHT_AMETHYST = hex_to_xterm("6000FF")
        cls.BRIGHT_MOSS = hex_to_xterm("20AA88")
        cls.BRIGHT_LILAC = hex_to_xterm("AA88EE")

        # Indivitual colors.
        cls.COLOR_BACKGROUND_RED = curses.pair_content(hex_to_xterm("660000"))[0]


@functools.cache
def get_color_pair(numb):
    curses.init_pair(numb, numb, -1)
    return curses.color_pair(numb)


@functools.cache
def hex_to_xterm(hex_code):
    # Convert hex code to RGB values
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    # Find the closest xterm-256color color value
    best_match = None
    min_distance = float("inf")
    for i in range(16, 232):  # Iterate over RGB colors (6x6x6 cube)
        # Calculate the RGB values for this xterm color
        xterm_r = ((i - 16) // 36) * 40 + 55
        xterm_g = (((i - 16) % 36) // 6) * 40 + 55
        xterm_b = ((i - 16) % 6) * 40 + 55
        # Calculate the distance between the two colors
        distance = ((r - xterm_r) ** 2 + (g - xterm_g) ** 2 + (b - xterm_b) ** 2) ** 0.5
        # Update the best match if this color is closer
        if distance < min_distance:
            min_distance = distance
            best_match = i
    for i in range(232, 256):  # Iterate over grayscale colors
        # Calculate the grayscale value for this xterm color
        xterm_gray = (i - 232) * 10 + 8
        # Calculate the distance between the two colors
        distance = (
            (r - xterm_gray) ** 2 + (g - xterm_gray) ** 2 + (b - xterm_gray) ** 2
        ) ** 0.5
        # Update the best match if this color is closer
        if distance < min_distance:
            min_distance = distance
            best_match = i
    return get_color_pair(best_match)


@functools.cache
def adjust_color(rgb_hex, adjustment):
    # Convert RGB hex to RGB values
    rgb_values = tuple(int(rgb_hex[i : i + 2], 16) for i in (0, 2, 4))

    # Adjust RGB values
    def _adjust_channel_value(channel_value):
        # Move towards black or white
        return int(max(0, min(255, channel_value + (255 - channel_value) * adjustment)))

    adjusted_rgb_values = tuple(
        _adjust_channel_value(channel_value) for channel_value in rgb_values
    )

    # Convert back to hex
    adjusted_rgb_hex = "{:02X}{:02X}{:02X}".format(*adjusted_rgb_values)

    return adjusted_rgb_hex


class CursesFormatter(TerminalFormatter):
    ansi_color_map = {
        30: "000000",  # Black
        31: "AA0000",  # Red
        32: "00AA00",  # Green
        33: "AAAA00",  # Yellow
        34: "0000AA",  # Blue
        35: "AA00AA",  # Magenta
        36: "00AAAA",  # Cyan
        37: "AAAAAA",  # White
        90: "707070",  # Black
        91: "FF0000",  # Red
        92: "00FF00",  # Green
        93: "FFFF00",  # Yellow
        94: "0000FF",  # Blue
        95: "FF00FF",  # Magenta
        96: "00FFFF",  # Cyan
        97: "FFFFFF",  # White
    }

    def __init__(self, **options):
        super(CursesFormatter, self).__init__(**options)

    def format_unencoded(self, tokensource, outfile):
        for ttype, value in tokensource:
            attr = self._get_curses_attr(ttype)
            if attr:
                yield (attr, value)
            else:
                yield (None, value)

    def _get_curses_attr(self, ttype):
        attr = 0
        if isinstance(ttype, AnsiEscapeAttrs):
            for sequ in ttype:
                if sequ.sequence_type == AnsiEscapeSequenceType.COLOR:
                    attr |= hex_to_xterm(self.ansi_color_map[sequ.code])
                elif sequ.sequence_type == AnsiEscapeSequenceType.C256:
                    attr |= get_color_pair(sequ.code)
                elif sequ.sequence_type == AnsiEscapeSequenceType.CRGB:
                    attr |= hex_to_xterm(sequ.code)
                elif sequ.sequence_type == AnsiEscapeSequenceType.REVERSE:
                    attr |= hex_to_xterm(self.ansi_color_map[sequ.code])
                    attr |= curses.A_REVERSE
                elif sequ.sequence_type == AnsiEscapeSequenceType.STYLE:
                    if sequ.code == 1:
                        attr |= curses.A_BOLD
                    elif sequ.code == 2:
                        attr |= curses.A_REVERSE
                    elif sequ.code == 3:
                        attr |= curses.A_ITALIC
                    elif sequ.code == 4:
                        attr |= curses.A_UNDERLINE
        else:
            try:
                style = self.style.style_for_token(ttype)
            except KeyError:
                return attr

            if SCROLLY_OPTS.fake_bold:
                color = style["color"]
                if style["bold"]:
                    if color:
                        attr |= hex_to_xterm(adjust_color(color, 0.5))
                    else:
                        attr |= hex_to_xterm("FFFFFF")
                elif color:
                    attr |= hex_to_xterm(color)
            else:
                if style["bold"]:
                    attr |= curses.A_BOLD
                if style["color"]:
                    attr |= hex_to_xterm(style["color"])

            if style["italic"]:
                attr |= curses.A_ITALIC

            if style["underline"]:
                attr |= curses.A_UNDERLINE

        return attr


class AnsiEscapeSequenceType:
    COLOR = "COLOR"
    REVERSE = "REVERSE"
    STYLE = "STYLE"
    CLEAR = "CLEAR"
    CRGB = "CRGB"
    C256 = "C256"


class AnsiEscapeSequence:
    def __init__(self, code, sequence_type):
        self.code = code
        self.sequence_type = sequence_type

    def __repr__(self):
        return f"AnsiEscapeSequence: code={self.code} type={self.sequence_type}"


class AnsiEscapeAttrs(list):
    pass


class AnsiEscapeLexer:
    color_codes = CursesFormatter.ansi_color_map.keys()
    ansi_escape_pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def get_tokens(self, text):
        start_index = 0
        end_index = 0
        current = AnsiEscapeAttrs()
        old_current = current.copy()

        def clear_color():
            for idx in reversed(range(len(current))):
                if current[idx].sequence_type != AnsiEscapeSequenceType.STYLE:
                    del current[idx]

        while end_index < len(text):
            if text[end_index] == "\x1b":
                end_index += 1
                if end_index == len(text):
                    continue
                if text[end_index] != "[":
                    continue
                end_index += 1
                if end_index == len(text):
                    continue
                sequence_start = end_index
                new_code = True
                while end_index < len(text):
                    char = text[end_index]
                    end_index += 1
                    if char == "m":
                        # Found a simple a sequence
                        try:
                            code = int(text[sequence_start : end_index - 1])
                        except ValueError:
                            continue
                        if code == 0:  # Reset
                            current.clear()
                            break
                        if code == 39:  # Reset forground
                            clear_color()
                            break
                        if code in self.color_codes:
                            clear_color()
                            current.append(
                                AnsiEscapeSequence(code, AnsiEscapeSequenceType.COLOR)
                            )
                            break
                        if (code + 10) in self.color_codes:  # High intensity
                            clear_color()
                            current.append(
                                AnsiEscapeSequence(
                                    code - 10, AnsiEscapeSequenceType.REVERSE
                                )
                            )
                            break
                        if 1 <= code <= 5:
                            current.append(
                                AnsiEscapeSequence(code, AnsiEscapeSequenceType.STYLE)
                            )
                            break

                    elif char == ";":
                        # Found a multi-part a sequence
                        sgr_parsed = self.parse_sgr(text[sequence_start - 2 :])
                        if sgr_parsed is None:
                            new_code = False
                            break
                        clear_color()
                        if sgr_parsed[0] == 5:
                            current.append(
                                AnsiEscapeSequence(
                                    sgr_parsed[1], AnsiEscapeSequenceType.C256
                                )
                            )
                        else:
                            current.append(
                                AnsiEscapeSequence(
                                    "".join(sgr_parsed[1:]), AnsiEscapeSequenceType.CRGB
                                )
                            )
                        while text[end_index] != "m":
                            end_index += 1
                        end_index += 1
                        break

                    elif not char.isdigit():
                        new_code = False
                        break
                else:
                    new_code = False

                if new_code:
                    cleaned = self.ansi_escape_pattern.sub(
                        "", text[start_index:end_index]
                    )
                    yield (AnsiEscapeAttrs(old_current), cleaned)
                    old_current = current.copy()
                    start_index = end_index
            else:
                end_index += 1
        cleaned = self.ansi_escape_pattern.sub("", text[start_index:end_index])
        yield (AnsiEscapeAttrs(old_current), cleaned)
        # This mimics the behaviour of pygments.
        return (AnsiEscapeAttrs([]), "")  # noqa

    def parse_sgr(self, text):
        if not text.startswith("\x1b[38;"):
            return None

        sequence = text[5:]
        end_index = sequence.find("m")

        if end_index == -1:
            return None

        for char in sequence[:end_index]:
            if not char.isdigit() and char != ";":
                raise ValueError([sequence, end_index])
                return None

        parts = sequence[:end_index].split(";")

        if len(parts) < 2:
            return None

        try:
            color_mode = int(parts[0])
            values = [int(x) for x in parts[1:]]
        except ValueError:
            return None

        if color_mode not in [2, 5]:
            return None

        if color_mode == 2 and len(values) != 3:
            return None

        if color_mode == 2 and any(x < 0 or x > 255 for x in values):
            return None

        if color_mode == 2:
            values = [f"{x:02X}" for x in values]
            values = "".join(values)

        values.insert(0, color_mode)

        return values


def parse_single_point_range(*args, **kwargs):
    index = parse_index_range(*args, **kwargs)
    if len(index) != 1:
        raise IndexError(f"Not a valid single point range {index}")
    return index[0]


def parse_index_range(message_str, max_count, current=None):
    """
    Parse a 'page-range' like string into a list of integer indices.

    :param message_str: The comma-delimited string specifying messages (e.g. "1, 2-3, 4-, -, 1-.").
    :param max_count: The maximum valid count; valid indices are 0..(max_count - 1).
    :return: A list of integers (including duplicates) corresponding to the expanded messages.
    :raises ValueError: If the format is invalid or indices fall outside 0..(max_count - 1).
    """

    # "-" is a special case for an empty input range.
    if max_count == 0 and message_str == "-":
        return []

    def cook_relative_truncations(input_str):
        pattern = re.compile(r"^([+~])(\d+)$")
        match = pattern.match(input_str)
        if match:
            prefix, rest = match.groups()
            if prefix == "+":
                return ".-." + input_str
            elif prefix == "~":
                return "." + input_str + "-."
        return input_str

    def expand_range(s, e):
        """Helper to expand [s..e] in ascending or descending order."""
        return list(range(s, e + (1 if s <= e else -1), 1 if s <= e else -1))

    def replace_relative_ranges(message_str):
        def replace_match(match):
            num1 = int(match.group(1))
            op = match.group(2)
            num2 = int(match.group(3))

            if op == "+":
                return str(num1 + num2)
            elif op == "~":
                return str(num1 - num2)

        pattern = r"(\d+)([+~])(\d+)"
        return re.sub(pattern, replace_match, message_str)

    if max_count <= 0:
        raise ValueError(f"max_count must be a positive integer, got {max_count}.")

    # Precook patterns like +10 to be relative
    message_str = cook_relative_truncations(message_str)

    # If current is set, we'll use it to replace '.' in the input string
    if current is not None:
        # Replace '.' with the current index
        message_str = message_str.replace(".", str(current))
    elif "." in message_str:
        raise ValueError(f"'.' in range where no current set: '{message_str}'")

    # Replace '$' with the last index
    message_str = message_str.replace("$", str(max_count - 1))
    # Compute range index offsets
    message_str = replace_relative_ranges(message_str)

    segments = (seg.strip() for seg in message_str.split(","))
    results = []

    for segment in segments:
        if not segment:
            # e.g. an empty segment if someone typed two commas in a row
            raise ValueError(f"Empty segment in input: '{message_str}'")

        if segment == "-":
            # Expand the entire range
            results.extend(range(0, max_count))
            continue

        dash_count = segment.count("-")

        if dash_count == 0:
            # Single integer
            try:
                idx = int(segment)
            except ValueError:
                raise ValueError(f"Invalid integer segment: '{segment}'")
            if idx < 0 or idx >= max_count:
                raise ValueError(
                    f"Index {idx} is outside valid range [0..{max_count - 1}]."
                )
            results.append(idx)

        elif dash_count == 1:
            # Could be "s-e", "s-", or "-e"
            parts = segment.split("-")
            if len(parts) != 2:
                raise ValueError(f"Malformed segment: '{segment}'")
            left, right = parts[0], parts[1]

            if left == "" and right == "":
                # This should not happen - see "-" handler above.
                raise ValueError("Invalid segment '-': missing start and end.")

            if left == "":
                # No start => 0..right
                try:
                    end = int(right)
                except ValueError:
                    raise ValueError(f"Invalid integer after dash: '{right}'")
                if not (0 <= end < max_count):
                    raise ValueError(
                        f"Index {end} is outside valid range [0..{max_count - 1}]."
                    )
                results.extend(expand_range(0, end))

            elif right == "":
                # No end => left..(max_count-1)
                try:
                    start = int(left)
                except ValueError:
                    raise ValueError(f"Invalid integer before dash: '{left}'")
                if not (0 <= start < max_count):
                    raise ValueError(
                        f"Index {start} is outside valid range [0..{max_count - 1}]."
                    )
                results.extend(expand_range(start, max_count - 1))

            else:
                # Full range => "s-e"
                try:
                    s = int(left)
                    e = int(right)
                except ValueError:
                    raise ValueError(f"Invalid integer(s) in segment: '{segment}'")
                if not (0 <= s < max_count):
                    raise ValueError(
                        f"Index {s} is outside valid range [0..{max_count - 1}]."
                    )
                if not (0 <= e < max_count):
                    raise ValueError(
                        f"Index {e} is outside valid range [0..{max_count - 1}]."
                    )
                results.extend(expand_range(s, e))

        else:
            # More than one dash => invalid
            raise ValueError(f"Malformed segment with multiple dashes: '{segment}'")

    if not results:
        raise ValueError("No indices found")

    return results


split_lines = re.compile("([\n])").split

join_lines = "".join


def get_lexer_by_name(name):
    lname = name.lower()
    if lname == "scl":
        return SCLLexer()
    elif lname == "ansiescape":
        return AnsiEscapeLexer()
    else:
        return lx_get_lexer_by_name(name)


@functools.lru_cache(maxsize=128)
def text_processor(text, width, lexer=MarkdownLexer):
    text = text.expandtabs(4)
    processed_lines = []

    blank_line = [(CursesDefs.MEDIUM_WHITE, "")]

    # Catch and elide empty strings.
    def append_text(tokens):
        processed_lines.append([token for token in tokens if token[1]])

    @functools.lru_cache(maxsize=128)
    def get_formatted(lexer, text, style):
        lexer = get_lexer_by_name(lexer)
        formatter = CursesFormatter(style=style)
        tokensource = lexer.get_tokens(text)
        highlighted_text = formatter.format_unencoded(tokensource, None)
        return [
            (attr, "".join(text for _, text in group))
            for attr, group in itertools.groupby(highlighted_text, key=lambda x: x[0])
        ]

    current_line = []
    current_len = 0
    for attr, values in get_formatted(lexer, text, get_formatter_style(lexer)):
        if attr is None:
            attr = 0
        for value in split_lines(values):
            if value == "\n":
                append_text(current_line)
                if current_len == width:
                    processed_lines.append(blank_line)
                current_line = []
                current_len = 0
            else:
                while current_len + len(value) > width:
                    chunk = width - current_len
                    lhs = value[:chunk]
                    value = value[chunk:]
                    current_line.append((attr, lhs))
                    append_text(current_line)
                    current_line = []
                    current_len = 0
                if value != "":
                    current_line.append((attr, value))
                    current_len += len(value)
    return processed_lines


def get_lexer_name(prompt):
    if prompt.syntax is not None:
        return prompt.syntax
    role = prompt.role
    if role == Role.BASH:
        return "bash"
    elif role == Role.SCL:
        return "scl"
    else:
        return "markdown"


def get_formatter_style(lexer):
    return SCROLLY_OPTS.style_overrides.get(lexer, SCROLLY_OPTS.style)


class ViewModes:
    SUMMARY_ALL = "SUMMARY_ALL"
    VIEW_LAST = "VIEW_LAST"
    VIEW_ALL = "VIEW_ALL"


SUMMARY_VIEW = ViewModes.SUMMARY_ALL


# Function to draw boxes around strings
def draw_boxes(stdscr, y_offset, prompts):
    curses.curs_set(0)
    height, width = stdscr.getmaxyx()
    if len(prompts) == 0:
        draw_pace(stdscr)
        return

    y = 0
    x = 0
    lexer = None
    for i, prompt in enumerate(prompts):
        if SUMMARY_VIEW == ViewModes.SUMMARY_ALL or (
            SUMMARY_VIEW == ViewModes.VIEW_LAST and i != len(prompts) - 1
        ):
            lexer = "markdown"
            raw_text = prompt.summary
            if raw_text is None:
                raw_text = "Awaiting summarisation"
        else:
            lexer = get_lexer_name(prompt)
            raw_text = prompt.text

        if prompt.filename:
            raw_text = f"FILE:`{prompt.filename}`\n\n{raw_text}"
        processed_text = text_processor(raw_text, width - 2, lexer)
        if y > height - y_offset:
            break
        header = f"{prompt.role}:{i}"
        colour = Role.get_role_colour(prompt.role)
        if SCROLLY_OPTS.fun:
            border_colour = colour
        else:
            border_colour = 0
        if i == 0:
            safe_addstr(
                stdscr,
                height,
                y + y_offset,
                x,
                Border.TOP_LEFT
                + Border.HORIZONTAL_DOUBLE * (width - 2)
                + Border.TOP_RIGHT,
                border_colour,
            )
        else:
            safe_addstr(
                stdscr,
                height,
                y + y_offset,
                x,
                Border.DOUBLE_TO_SINGLE_LEFT
                + Border.HORIZONTAL_DOUBLE * (width - 2)
                + Border.DOUBLE_TO_SINGLE_RIGHT,
                border_colour,
            )
        safe_addstr(
            stdscr,
            height,
            y + y_offset,
            x + width // 2 - len(header) // 2,
            header,
            colour | curses.A_BOLD,
        )

        for line in processed_text:
            safe_addstr(
                stdscr, height, y + 1 + y_offset, x, Border.VERTICAL, border_colour
            )
            across = x + 1
            for attr, value in line:
                safe_addstr(stdscr, height, y + 1 + y_offset, across, value, attr)
                across += len(value)
            safe_addstr(
                stdscr,
                height,
                y + 1 + y_offset,
                x + width - 1,
                Border.VERTICAL,
                border_colour,
            )
            y += 1
        y += 1

    safe_addstr(
        stdscr,
        height,
        y + y_offset,
        x,
        Border.BOTTOM_LEFT
        + Border.HORIZONTAL_DOUBLE * (width - 2)
        + Border.BOTTOM_RIGHT,
        border_colour,
    )
    stdscr.move(height - 1, 0)


ESCAPE = object()


def show_command_panel(stdscr, message, single_char=False, win_width=40, win_height=10):
    height, width = stdscr.getmaxyx()
    if win_width > width:
        raise ValueError("Command pannel too wide for screen")
    if win_height > height:
        raise ValueError("Command pannel too tall for screen")
    curses.curs_set(1)
    win_y = (height - win_height) // 2
    win_x = (width - win_width) // 2

    if message.startswith("**"):
        color = CursesDefs.BRIGHT_RED
    elif single_char:
        color = CursesDefs.BRIGHT_MAGENTA
    else:
        color = CursesDefs.BRIGHT_GREEN

    win = curses.newwin(win_height, win_width, win_y, win_x)
    response = ""
    cursor_position = 0

    def draw_message():
        nonlocal win
        h, w = win.getmaxyx()
        y = 0
        for line in text_processor(message, w - 2, "markdown"):
            across = 0
            for attr, value in line:
                safe_addstr(win, h - 1, y + 1, across + 1, value, attr)
                across += len(value)
            y += 1
        return y

    while True:
        win.erase()
        win.attron(color)
        try:
            win.border()
        finally:
            win.attroff(color)
        y = draw_message()
        x = 1  # Left border
        y += 2  # Header
        r = win_width - 2  # RHS
        m = (r - 2) * win_height - y  # window max chars.
        s = max(0, len(response) - m)  # Start of response to view
        win.move(y, x)
        for i, c in enumerate(response[s:]):
            if i == cursor_position - s:
                win.addstr(y, x, c, curses.A_REVERSE)
            else:
                win.addstr(y, x, c)
            x += 1
            if x >= r:
                y += 1
                x = 2
                if y > win_height - 2:
                    break

        win.refresh()
        c = stdscr.getch()
        if c == ord("\n"):
            break
        elif c in (curses.KEY_BACKSPACE, curses.KEY_DC, 127):
            if cursor_position > 0:
                response = response[: cursor_position - 1] + response[cursor_position:]
                cursor_position -= 1
        elif c == KEY_ESCAPE or c == curses.KEY_RESIZE:
            response = ESCAPE
            break
        elif c == curses.KEY_LEFT:
            cursor_position = max(0, cursor_position - 1)
        elif c == curses.KEY_RIGHT:
            cursor_position = min(len(response), cursor_position + 1)
        else:
            # Random times a high value gets through here.
            try:
                vc = chr(c)
            except ValueError:
                continue
            if single_char:
                return vc
            else:
                response = response[:cursor_position] + vc + response[cursor_position:]
                cursor_position += 1

    del win
    stdscr.touchwin()  # mark the entire window as needing to be redrawn
    stdscr.refresh()  # redraw the window

    curses.curs_set(0)
    return response


@contextlib.contextmanager
def signal_handler(sig, handler):
    original_handler = signal.signal(sig, handler)
    try:
        yield
    finally:
        signal.signal(sig, original_handler)


def execute_bash_script(script, input_string=None):
    script_file_name = None
    output_file_name = None
    output = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".sh"
        ) as script_file:
            script_file.write(script)
            script_file_name = script_file.name

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_out:
            output_file_name = tmp_out.name

        with utf8_open(output_file_name, "a") as out_handle:
            if input_string is not None:
                env = os.environ.copy()
                env["TERM"] = "xterm-256color"
                env["COLORTERM"] = "256color"
                proc = subprocess.Popen(
                    ["bash", script_file_name],
                    stdin=subprocess.PIPE,
                    stdout=out_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                input_bytes = input_string.encode("utf-8")
                proc.stdin.write(input_bytes)
                proc.stdin.close()
            else:
                proc = subprocess.Popen(
                    ["bash", script_file_name],
                    stdin=subprocess.DEVNULL,
                    stdout=out_handle,
                    stderr=subprocess.STDOUT,
                )

            # Function to handle SIGINT, send it to the subprocess
            def handle_sigint(signum, frame):
                if proc is not None:
                    proc.send_signal(signal.SIGINT)

            with signal_handler(signal.SIGINT, handle_sigint):
                with utf8_open(output_file_name, "r") as monitor:
                    last_pos = 0
                    while proc.poll() is None:
                        try:
                            current_size = os.path.getsize(output_file_name)
                        except OSError as e:
                            proc.kill()
                            raise e
                        if current_size > last_pos:
                            monitor.seek(last_pos)
                            new_data = monitor.read()
                            print(new_data)
                            output.append(new_data)
                            last_pos = monitor.tell()
                        time.sleep(0.1)
                    monitor.seek(last_pos)
                    new_data = monitor.read()
                    print(new_data)
                    output.append(new_data)

    finally:
        # In theory a script can delete its own input and output files.
        if script_file_name and os.path.exists(script_file_name):
            os.remove(script_file_name)
        if output_file_name and os.path.exists(output_file_name):
            os.remove(output_file_name)

    return "".join(output)


def handle_scroll(stdscr, prompts, scroll_offset):
    stdscr.erase()
    draw_boxes(stdscr, scroll_offset, prompts)
    stdscr.refresh()


def handle_quit(stdscr):
    try:
        pressed = show_command_panel(
            stdscr,
            """**Do you really want to quit?**
* `Q` or `q` to quit
* Any other key to return
""",
            True,
        )
        if pressed is ESCAPE:
            return None
        return pressed.upper() != "Q"
    except KeyboardInterrupt:
        return handle_quit(stdscr)


def do_ask_role(stdscr):
    prmpt = """## Enter role
- U: User, X: Cooked, A: AI,
- S: System, C: Comment,
- T: Title, B: Bash,
- L: SCL, G: Generator"""
    while True:
        role_choice = show_command_panel(stdscr, prmpt, True)
        if role_choice is ESCAPE:
            return ESCAPE
        role_choice = role_choice.upper()
        if role_choice == "U":
            role = Role.USER
            break
        if role_choice == "X":
            role = Role.COOKED
            break
        elif role_choice == "A":
            role = Role.AI
            break
        elif role_choice == "S":
            role = Role.SYSTEM
            break
        elif role_choice == "B":
            role = Role.BASH
            break
        elif role_choice == "C":
            role = Role.COMMENT
            break
        elif role_choice == "T":
            role = Role.TITLE
            break
        elif role_choice == "L":
            role = Role.SCL
            break
        elif role_choice == "G":
            role = Role.GENERATOR
            break
        prmpt = "Enter valid role\n U, X, A, S, C, T, B, L, G"
    return role


def handle_read_from_stdin(stdscr, prompts):
    with curses_step_out(stdscr):
        text = read_from_stdin()
    text = clean_text(text)
    prompts.append(Prompt(Role.USER, text, syntax="markdown"))
    background_summarise_prompt(prompts[-1])


def handle_new(stdscr, prompts):
    role = do_ask_role(stdscr)
    if role is ESCAPE:
        return
    editor = SimpleEditor(stdscr, text="", lexer=get_lexer_name(Prompt(role, "")))
    do_new(prompts, role, editor.edit(), editor)


def do_new(prompts, role, text, editor=None, syntax=None):
    text = clean_text(text)
    prompts.append(Prompt(role, text, editor=editor, syntax=syntax))
    background_summarise_prompt(prompts[-1])


def handle_execute(stdscr, prompts):
    stdscr.clear()
    stdscr.refresh()
    cooked = prompts[-1].role == Role.COOKED
    new_text = asyncio.run(handle_execute_async(stdscr, prompts))
    prompts.append(Prompt(Role.AI, new_text))
    background_summarise_prompt(prompts[-1])
    if cooked:
        stdscr.clear()
        prompts.extend(SystemPrompts.UNCOOKED)
        new_text = asyncio.run(handle_execute_async(stdscr, prompts))
        prompts.append(Prompt(Role.AI, new_text))
        background_summarise_prompt(prompts[-1])
    stdscr.clear()
    stdscr.refresh()


def handle_role(stdscr, prompts):
    message_index = show_command_panel(stdscr, "## Enter message range to change:")
    if message_index is ESCAPE:
        return
    message_index = parse_index_range(message_index, len(prompts))

    role = do_ask_role(stdscr)
    if role is ESCAPE:
        return
    for index in message_index:
        prompts[index].role = role


def do_scl_read(stdscr, absname, prompts):
    with utf8_open(absname, "r") as f:
        if not SCLRecorder.try_load(stdscr, f.read()):
            return False
    new_prompts, new_scroll_offset = Status.conversations.get_current_conversation()
    prompts.clear()
    prompts.extend(new_prompts)
    return True


def do_load(stdscr, filename, prompts):
    if filename is ESCAPE:
        return
    absname = create_file_path(SCROLLY_OPTS.prompts_dir, filename)
    Status.last_loaded = absname
    if do_scl_read(stdscr, absname, prompts):
        return

    # Legacy json support.
    with utf8_open(absname, "r") as f:
        data = json.load(f)
    prompts.clear()
    for item in data:
        prompts.append(
            Prompt(
                role=item["role"],
                text=item["text"],
                filename=item.get("filename"),
                syntax=item.get("syntax"),
                summary=item.get("summary"),
            )
        )
    Status.conversations.set_current_conversation(
        Conversation(prompts=prompts, scroll_offset=0)
    )


def do_read(stdscr, filename, prompts):
    if filename is ESCAPE:
        return
    absname = create_file_path(SCROLLY_OPTS.prompts_dir, filename)
    Status.last_read = absname
    if do_scl_read(stdscr, absname, prompts):
        return

    # Legacy json support.
    with utf8_open(absname, "r") as f:
        data = json.load(f)
    prompts.clear()
    for index, convo_data in enumerate(data):
        convo = []
        for item in convo_data:
            convo.append(
                Prompt(
                    role=item["role"],
                    text=item["text"],
                    filename=item.get("filename"),
                    syntax=item.get("syntax"),
                    summary=item.get("summary"),
                )
            )

        Status.conversations.set_conversation(
            index, Conversation(prompts=convo, scroll_offset=0)
        )
    Status.conversations.set_current_index(0)
    prompts.clear()
    prompts.extend(Status.conversations.get_current_conversation()[0])


def handle_load(stdscr, prompts):
    do_load(stdscr, show_command_panel(stdscr, "## Enter filename to load:"), prompts)


def handle_read(stdscr, prompts):
    do_read(stdscr, show_command_panel(stdscr, "## Enter filename to read:"), prompts)


def handle_append(stdscr, prompts):
    new_prompts = []
    do_load(
        show_command_panel(stdscr, "## Enter filename to append:"),
        new_prompts,
    )
    prompts.extend(new_prompts)


def handle_yank(stdscr, prompts):
    range_def = show_command_panel(stdscr, "## Enter range to yank:")
    if range_def is ESCAPE:
        return
    index_range = parse_index_range(range_def, len(prompts))
    Status.yanked.set_current([prompts[index] for index in index_range])


def handle_undelete(stdscr, prompts):
    range_def = show_command_panel(stdscr, "## Enter range to undelete:")
    if range_def is ESCAPE:
        return
    for prompt in Status.conversations.undelete(range_def):
        prompts.append(prompt)


def handle_unyank(stdscr, prompts):
    yanked = Status.yanked.get_current()
    if yanked is not None:
        for y in yanked:
            prompts.append(y.copy())
            background_summarise_prompt(prompts[-1])


def handle_paste(stdscr, prompts):
    paste_id = show_command_panel(stdscr, "## Enter paste number:")
    if paste_id is ESCAPE:
        return
    if paste_id and paste_id.upper()[0] == "P":
        paste_id = paste_id[1:]
    text = load_paste(paste_id)
    prompts.append(Prompt(Role.USER, text, None))


class ReloadOptions:
    RELOAD = "r"
    OVER_WRITE = "o"
    RUN_AWAY = "x"
    VALID = set("rox")

    @classmethod
    def is_valid(cls, option):
        return option is not None and option.lower() in cls.VALID


def ask_if_reload(stdscr):
    reload = None
    while not ReloadOptions.is_valid(reload):
        reload = show_command_panel(
            stdscr,
            "**File Contents Changed**\n* r: reload\n* o: overwrite\n* x: get out of here",
            True,
        )
        if reload is ESCAPE:
            return ReloadOptions.RUN_AWAY
    return reload.lower()


def handle_edit(stdscr, prompts):
    message_index = show_command_panel(stdscr, "## Enter message number to edit:")

    if message_index is ESCAPE:
        return

    message_index = parse_single_point_range(message_index, len(prompts))
    prompt = prompts[message_index]
    if prompt.filename is not None:
        with utf8_open(prompt.filename, "r") as f:
            file_text = f.read()
        if file_text != prompt.text:
            reload = ask_if_reload(stdscr)
            if reload == ReloadOptions.RUN_AWAY:
                return
            if reload == ReloadOptions.RELOAD:
                prompt.text = file_text
                prompt.editor = None
            if reload == ReloadOptions.OVER_WRITE:
                with utf8_open(prompt.filename, "w") as f:
                    f.write(prompt.text)

    if prompt.editor is None:
        if prompt.filename is not None:
            prompt.editor = SimpleEditor(
                stdscr,
                text=prompt.text,
                lexer=get_lexer_name(prompt),
                filename=prompt.filename,
            )
        else:
            prompt.editor = SimpleEditor(
                stdscr, text=prompt.text, lexer=get_lexer_name(prompt)
            )
    else:
        prompt.editor.set_lexer(get_lexer_name(prompt))

    new_text = prompt.editor.edit()
    if prompt.filename is not None:
        with utf8_open(prompt.filename, "r") as f:
            file_text = f.read()
        if file_text != prompt.text and file_text != prompt.editor.original_text:
            reload = ask_if_reload(stdscr)
            if reload == ReloadOptions.RUN_AWAY:
                return
            elif reload == ReloadOptions.OVER_WRITE:
                prompt.text = new_text
                with utf8_open(prompt.filename, "w") as f:
                    f.write(new_text)
            elif reload == ReloadOptions.RELOAD:
                prompt.text = file_text
            prompt.editor.original_text = prompt.text
        elif new_text != prompt.text:
            prompt.text = new_text
            if new_text != file_text:
                with utf8_open(prompt.filename, "w") as f:
                    f.write(new_text)
        background_summarise_prompt(prompt)
    else:
        prompt.text = new_text


def handle_delete(stdscr, prompts):
    range_def = show_command_panel(stdscr, "## Enter range delete:")
    if range_def is ESCAPE:
        return
    index_range = sorted(set(parse_index_range(range_def, len(prompts))))
    for idx in index_range:
        Status.conversations.add_deleted(prompts[idx])
    for idx in reversed(index_range):
        del prompts[idx]


def handle_slot(prompts, scroll_offset, index):
    Status.conversations.set_current_conversation(
        Conversation(prompts=prompts, scroll_offset=scroll_offset)
    )
    Status.conversations.set_current_index(index)
    new_prompts, new_scroll_offset = Status.conversations.get_current_conversation()
    prompts.clear()
    prompts.extend(new_prompts)
    return new_scroll_offset


def clean_text(input_text):
    """Clean text by removing trailing whitespace."""
    return "\n".join(line.rstrip() for line in input_text.split("\n"))


def clear_screen():
    """Send the escape code to clear the screen, outsidecurses."""
    print("\033[2J\033[H", end="", flush=True)


def read_from_stdin():
    """
    Reads from stdin and returns the text read.
    Stops reading from stdin when Ctrl+D is encountered.
    Prints out what is being read in as it comes in.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        text = []
        char = ""
        line = []
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if sys.stdin in rlist:
                char = sys.stdin.read(1)
                if char == "\x04":
                    text.append("".join(line))
                    break
                line.append(char)
                if char in "\n\r":
                    block = "".join(line)
                    line.clear()
                    sys.stdout.write(block)
                    if char == "\r":
                        sys.stdout.write("\n")
                    sys.stdout.flush()
                    text.append(block)
            elif line:
                block = "".join(line)
                line.clear()
                sys.stdout.write(block)
                sys.stdout.flush()
                text.append(block)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return "".join(text)


@contextlib.contextmanager
def curses_step_out(stdscr):
    """
    A context manager that steps out of curses mode on entry and restarts it on exit.
    """
    curses_stop(stdscr)
    clear_screen()
    try:
        yield
    finally:
        curses_restart(stdscr)


def curses_stop(stdscr):
    """
    Stops the curses mode, resets the shell mode, and clears the screen.
    """
    stdscr.clear()
    stdscr.move(0, 0)
    stdscr.refresh()
    height, _ = stdscr.getmaxyx()
    curses.reset_shell_mode()
    curses.endwin()


def curses_restart(stdscr):
    """
    Restarts the curses mode, resets the program mode, and clears the screen.
    """
    curses.initscr()
    curses.reset_prog_mode()
    stdscr.clear()
    stdscr.move(0, 0)
    stdscr.refresh()


def handle_info(stdscr):
    show_command_panel(
        stdscr,
        f"""## Current Status
* Slot: {Status.conversations.get_current_index()}""",
        True,
    )


def handle_view(stdscr, prompts):
    message_index = show_command_panel(stdscr, "## Enter message number to view:")

    if message_index is ESCAPE:
        return

    message_index = parse_single_point_range(message_index, len(prompts))
    prompt = prompts[message_index]
    lexer = prompt.syntax if prompt.syntax else "markdown"
    dline = "=" * 40
    sline = "-" * 40
    with curses_step_out(stdscr):
        print(f"\n{dline}")
        print(
            f"""
Role:   {prompt.role}
Syntax: {prompt.syntax}
File:   {prompt.filename}
""".strip()
        )
        print(f"{sline}\n")
        print_highlighted(prompt.text, lexer)
        input(f"\n{dline}\nPress Enter to continue...")


def send_paste(content, title, stdscr):
    pste = create_paste(content, f"Scrolly: {title}_{time.monotonic()}")
    msg = f"""## URL:
* `{pste.strip('/').split('/')[-1]}`"""
    show_command_panel(stdscr, msg, True)


def handle_copy(stdscr, prompts):
    message_index = show_command_panel(stdscr, "## Enter message number to paste:")
    if message_index is ESCAPE:
        return

    title = show_command_panel(stdscr, "## Enter title for paste:")
    if title is ESCAPE:
        return

    message_index = parse_single_point_range(message_index, len(prompts))
    send_paste(clean_text(prompts[message_index].text), title, stdscr)


def handle_bash(stdscr, prompts, stdin=None):
    prompt = prompts[-1]
    if prompt.role != Role.BASH:
        change = show_command_panel(
            stdscr,
            f"""**Not executable**
* Role {prompt.role} is not executable.
* Please read it and check it is safe.
* Hit X to convert and execute.
* Hit any other key get out.""",
            True,
        )
        if change != "X":
            return
        prompt.role = Role.BASH
    to_execute = extract_bash_script(prompt.text)
    if to_execute is not prompt.text:
        prompts.append(Prompt(role=Role.BASH, text=to_execute, syntax="bash"))
        prompt.role = Role.COMMENT
        prompt = prompts[-1]
    with curses_step_out(stdscr):
        print("Execution Bash")
        print("==============\n")
        text = execute_bash_script(prompt.text, stdin)
    prompts.append(Prompt(role=Role.USER, text=text, syntax="ansiescape"))
    background_summarise_prompt(prompts[-1])


def handle_automate(stdscr, prompts, scroll_offset):
    prompt = prompts[-1]
    if prompt.role != Role.SCL:
        change = show_command_panel(
            stdscr,
            f"""**Not Automatable**
* Role {prompt.role} is not automatable.
* Please read it and check it is safe.
* Hit T to convert and automatable.
* Hit any other key get out.""",
            True,
        )
        if change != "T":
            return
        prompt.role = Role.SCL
    to_automate = prompt.text
    automator = SCLAutomator(stdscr, prompts, scroll_offset)
    try:
        automator.execute_statements(to_automate)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        handle_exception(stdscr, e)
    finally:
        new_prompts, new_scroll_offset = Status.conversations.get_current_conversation()
        prompts.clear()
        prompts.extend(new_prompts)
    return new_scroll_offset


def dispatch_yank_to_bash(stdscr, prompts, method):
    yanked = Status.yanked.get_current()
    if not check_yank(stdscr, yanked):
        return
    yanked = "\n".join(prompt.text for prompt in Status.yanked.get_current())
    return method(stdscr, prompts, yanked)


def handle_reyank_to_bash(stdscr, prompts):
    dispatch_yank_to_bash(stdscr, prompts, handle_rerun_bash)


def handle_yank_to_bash(stdscr, prompts):
    dispatch_yank_to_bash(stdscr, prompts, handle_bash)


def handle_rerun_bash(stdscr, prompts, stdin=None):
    found = None
    for index, prompt in reversed(list(enumerate(prompts))):
        if prompt.role == Role.BASH:
            found = index, prompt
            break
    if found is None:
        show_command_panel(stdscr, "**Nothing to run**", True)
        return
    del prompts[found[0]]
    prompts.append(found[1])
    handle_bash(stdscr, prompts, stdin)


def handle_copy_all(stdscr, prompts):
    title = show_command_panel(stdscr, "## Enter title for paste:")
    if title is ESCAPE:
        return
    scl = SCLRecorder.conversation_to_scl(prompts, header=True)
    send_paste("\n".join(scl), title, stdscr)


def do_save(filename, prompts, scroll_offset, all_slots=False):
    """Save AI conversation as JSON file"""
    if filename is ESCAPE:
        return

    absname = create_file_path(SCROLLY_OPTS.prompts_dir, filename)

    if os.path.exists(absname) and os.path.getsize(absname) > 0:
        backup_filename = f"{absname}.bak"
        os.replace(absname, backup_filename)

    Status.conversations.set_current_conversation(
        Conversation(prompts=prompts, scroll_offset=scroll_offset)
    )

    if all_slots:
        scl = SCLRecorder.conversations_to_scl(header=True)
        with utf8_open(absname, "w") as f:
            f.write("\n".join(scl))
        Status.last_written = absname
    else:
        scl = SCLRecorder.conversation_to_scl(prompts, header=True)
        with utf8_open(absname, "w") as f:
            f.write("\n".join(scl))
        Status.last_saved = absname


def handle_save(stdscr, prompts, scroll_offset):
    do_save(
        show_command_panel(stdscr, "## Enter filename to save as:"),
        prompts,
        scroll_offset,
    )


def handle_write(stdscr, prompts, scroll_offset):
    do_save(
        show_command_panel(stdscr, "## Enter filename to write as:"),
        prompts,
        scroll_offset,
    )


def auto_save(prompts, scroll_offset):
    if SCROLLY_OPTS.auto_save is not None:
        do_save(SCROLLY_OPTS.auto_save, prompts, scroll_offset, all_slots=True)


def handle_move(stdscr, prompts):
    message_index = show_command_panel(stdscr, "## Enter message number to move:")
    if message_index is ESCAPE:
        return

    message_index = parse_single_point_range(message_index, len(prompts))
    new_index = show_command_panel(stdscr, "## Enter new position:")
    if new_index is ESCAPE:
        return
    new_index = parse_single_point_range(new_index, len(prompts))
    prompts.insert(new_index, prompts.pop(message_index))


def handle_syntax(stdscr, prompts):
    message_index = show_command_panel(stdscr, "## Enter message number to highlight:")
    if message_index is ESCAPE:
        return

    message_index = parse_single_point_range(message_index, len(prompts))
    name = show_command_panel(stdscr, "## Enter highlighter name:")
    if name is ESCAPE:
        return
    # Is it a thing?
    try:
        get_lexer_by_name(name)
    except ValueError:
        raise ValueError("Not known - see: https://pygments.org/docs/lexers/")
    prompts[message_index].syntax = name


def handle_exception(stdscr, e):
    with curses_step_out(stdscr):
        print("Woopsy-Doodle")
        print("=============\n")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        print("\nPress Enter to continue...")
        input()


def handle_up(scroll_offset, count=1):
    return scroll_offset - count


def handle_down(stdscr, scroll_offset, count=1):
    height, _ = stdscr.getmaxyx()
    return min(scroll_offset + count, height - 1)


def scroll_page(stdscr, prompts, scroll_offset, action):
    if not prompts:
        return 0
    try:
        old_offset = scroll_offset
        stdscr.timeout(0)
        tlc_chars = Border.DOUBLE_TO_SINGLE_LEFT + Border.TOP_LEFT
        new_offset = action(scroll_offset)
        while new_offset != scroll_offset:
            c = stdscr.getch()
            if c != -1:
                curses.ungetch(c)
                break
            scroll_offset = new_offset
            height, width = stdscr.getmaxyx()
            handle_scroll(stdscr, prompts, scroll_offset)
            top_line = stdscr.instr(0, 0, width).decode("utf-8")
            if top_line.strip() == "":
                return old_offset
            top_left_corner = top_line[0]
            if top_left_corner in tlc_chars:
                break
            new_offset = action(scroll_offset)
        return scroll_offset
    finally:
        stdscr.timeout(-1)


def handle_up_page(stdscr, prompts, scroll_offset):
    return scroll_page(stdscr, prompts, scroll_offset, handle_up)


def handle_down_page(stdscr, prompts, scroll_offset):
    return scroll_page(stdscr, prompts, scroll_offset, lambda x: handle_down(stdscr, x))


def handle_scroll_to_bottom(stdscr, prompts, scroll_offset):
    if not prompts:
        return 0
    while True:
        height, width = stdscr.getmaxyx()
        handle_scroll(stdscr, prompts, scroll_offset)
        bottom_line = stdscr.instr(height - 1, 0, width).decode("utf-8")
        if bottom_line.strip() == "":
            break
        scroll_offset = handle_up(scroll_offset, stdscr.getmaxyx()[0])

    while True:
        height, width = stdscr.getmaxyx()
        handle_scroll(stdscr, prompts, scroll_offset)
        bottom_line = stdscr.instr(height - 1, 0, width).decode("utf-8")
        if bottom_line.strip() != "":
            break
        scroll_offset = handle_down(stdscr, scroll_offset)

    return scroll_offset


def handle_scroll_to_top(stdscr, prompts, scroll_offset):
    if not prompts:
        return 0
    while True:
        height, width = stdscr.getmaxyx()
        handle_scroll(stdscr, prompts, scroll_offset)
        top_line = stdscr.instr(0, 0, width).decode("utf-8")
        if top_line.strip() == "":
            break
        scroll_offset = handle_down(stdscr, scroll_offset, stdscr.getmaxyx()[0])

    return 0


class FileBrowser:
    def __init__(self, stdscr, current_dir):
        self.current_dir = current_dir
        self.stdscr = stdscr
        self.scroll_pos = 0
        self.columns = 2
        self.cursor_pos = 0
        self.search_mode = False
        self.search_query = ""

    def load_file_list(self):
        self.files = sorted([".."] + os.listdir(self.current_dir))
        self.cursor_pos = 0
        self.scroll_pos = 0

    def up_one(self):
        self.cursor_pos = max(0, self.cursor_pos - 1)
        if self.cursor_pos < self.scroll_pos:
            self.scroll_pos -= 1

    def down_one(self):
        self.cursor_pos = min(len(self.files) - 1, self.cursor_pos + 1)
        if self.cursor_pos >= self.scroll_pos + self.get_visible_rows():
            self.scroll_pos += 1

    def find(self, name, prefix=False):
        old_sp = self.scroll_pos
        old_cp = self.cursor_pos
        self.scroll_pos = 0
        self.cursor_pos = 0
        for filename in self.files:
            self.draw_screen()
            if name == filename or (prefix and filename.startswith(name)):
                return filename
            self.down_one()
        self.scroll_pos = old_sp
        self.scroll_pos = old_cp
        self.draw_screen()
        return None

    def run(self):
        self.load_file_list()

        while True:
            try:
                self.draw_screen()
                c = self.stdscr.getch()
                if c == ord("q") or c == KEY_ESCAPE:
                    break
                if c == curses.KEY_RESIZE:
                    selected_file = self.files[self.cursor_pos]
                    self.load_file_list()
                    self.find(selected_file)
                elif c == curses.KEY_LEFT:
                    self.search_mode = False
                    self.up_one()
                elif c == curses.KEY_UP:
                    self.search_mode = False
                    for _ in range(10):
                        self.draw_screen()
                        self.up_one()
                elif c == curses.KEY_RIGHT:
                    self.search_mode = False
                    self.down_one()
                elif c == curses.KEY_DOWN:
                    self.search_mode = False
                    for _ in range(10):
                        self.draw_screen()
                        self.down_one()
                elif c == ord("\n"):
                    self.search_mode = False
                    selected_file = self.files[self.cursor_pos]
                    if selected_file == "..":
                        # Go up one directory level
                        parent_dir = os.path.dirname(self.current_dir)
                        if parent_dir != self.current_dir:  # Check if we're not at root
                            self.current_dir = parent_dir
                            self.load_file_list()
                            self.scroll_pos = 0
                        else:
                            # We're already at root, do nothing
                            pass
                    elif os.path.isdir(os.path.join(self.current_dir, selected_file)):
                        # Selected file is a directory, navigate into it
                        self.current_dir = os.path.join(self.current_dir, selected_file)
                        self.load_file_list()
                        self.scroll_pos = 0
                    else:
                        return self.files[self.cursor_pos]
                elif c == ord("/"):
                    self.search_mode = not self.search_mode
                    self.search_query = ""
                elif c in (curses.KEY_BACKSPACE, curses.KEY_DC):
                    if self.search_mode:
                        self.search_query = self.search_query[:-1]
                elif self.search_mode and c >= 32:
                    if self.search_mode:
                        self.search_query += chr(c)
                        self.find_best_match()
                else:
                    if c == ord("n"):
                        self.new_file()
                    elif c == ord("N"):
                        self.new_dir()
                    elif c == ord("d"):
                        self.delete_file()
                    elif c == ord("!"):
                        self.shell()
            except curses.error:
                self.stdscr.clear()
                self.draw_screen()

    def find_best_match(self):
        best_match_index = -1
        for i, filename in enumerate(self.files):
            if filename.startswith(self.search_query):
                best_match_index = i
                break
        if best_match_index != -1:
            self.cursor_pos = best_match_index
            self.scroll_pos = max(0, self.cursor_pos - self.get_visible_rows() + 1)

    def draw_screen(self):
        curses.curs_set(0)
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        column_width = width // self.columns
        self.stdscr.addstr(0, 0, "Current dir: " + self.current_dir)
        for i in range(self.get_visible_rows()):
            file_index = self.scroll_pos + i
            if file_index < len(self.files):
                file = self.files[file_index]
                color = CursesDefs.WHITE
                if os.path.isdir(os.path.join(self.current_dir, file)):
                    color = CursesDefs.CYAN
                x = (i % self.columns) * column_width
                y = i // self.columns + 1
                if file_index == self.cursor_pos:
                    self.stdscr.addstr(
                        y, x, file[: column_width - 1], color | curses.A_REVERSE
                    )
                else:
                    self.stdscr.addstr(y, x, file[: column_width - 1], color)
        if self.search_mode:
            self.stdscr.addstr(
                height - 4, 0, Border.HORIZONTAL_DOUBLE * width, CursesDefs.BRIGHT_GREEN
            )
            self.stdscr.addstr(height - 3, 0, self.search_query, CursesDefs.CYAN)
        else:
            self.stdscr.addstr(
                height - 4, 0, Border.HORIZONTAL_DOUBLE * width, CursesDefs.BRIGHT_WHITE
            )
            self.stdscr.addstr(
                height - 3,
                0,
                "left/right, down/up: move, /: isearch on|off",
                CursesDefs.YELLOW,
            )
            self.stdscr.addstr(
                height - 2, 0, "n: new file, N:new dir", CursesDefs.YELLOW
            )
            self.stdscr.addstr(
                height - 1, 0, "d: delete file, !: shell, q: quit", CursesDefs.YELLOW
            )
        self.stdscr.refresh()

    def shell(self):
        with curses_step_out(self.stdscr):
            subprocess.run([SCROLLY_OPTS.shell], cwd=self.current_dir)
        self.load_file_list()

    def get_visible_rows(self):
        height, _ = self.stdscr.getmaxyx()
        return (height - 5) * self.columns

    def new_file(self):
        height, width = self.stdscr.getmaxyx()
        filename = show_command_panel(self.stdscr, "## Enter new filename:")
        with utf8_open(os.path.join(self.current_dir, filename), "w"):
            pass
        self.load_file_list()
        self.find(filename)

    def new_dir(self):
        height, width = self.stdscr.getmaxyx()
        dirname = show_command_panel(self.stdscr, "## Enter new dirname:")
        os.mkdir(os.path.join(self.current_dir, dirname))
        self.load_file_list()
        self.find(dirname)

    def delete_file(self):
        filename = self.files[self.cursor_pos]
        height, width = self.stdscr.getmaxyx()
        yn = show_command_panel(
            self.stdscr, f"## Delete {filename}?\n* Press `y` or `n`"
        )
        if yn == "y":
            os.remove(os.path.join(self.current_dir, filename))
            self.load_file_list()

        if self.cursor_pos < self.scroll_pos:
            self.scroll_pos -= 1


# Make the current directory for different actions stateful.
# Other statefulness (previous dir, next dir etc) might be added.
class FileBrowsers:
    _browsers = {}

    @classmethod
    def _load(cls, browser_type, stdscr):
        cls._browsers[browser_type] = cls._browsers.get(browser_type) or FileBrowser(
            stdscr, SCROLLY_OPTS.prompts_dir
        )
        return cls._browsers[browser_type]

    @classmethod
    def append_browser(cls, stdscr):
        return cls._load("append", stdscr)

    @classmethod
    def command_browser(cls, stdscr):
        return cls._load("command", stdscr)

    @classmethod
    def file_browser(cls, stdscr):
        return cls._load("file", stdscr)

    @classmethod
    def load_browser(cls, stdscr):
        return cls._load("load", stdscr)

    @classmethod
    def save_browser(cls, stdscr):
        return cls._load("save", stdscr)


def ctrl_key_code(char):
    # 64 and 96 are just magical numbers terminals use to map control keys.
    # A is 65 and a is 97 if that helps?
    if char.isalpha():
        ctrl_key_code = ord(char) - (64 if char.isupper() else 96)
    else:
        ctrl_key_code = ord(char) - 64
    return ctrl_key_code


def double_column_list(input_list):
    # If the input list has an odd number of rows, add two blank elements to the last row
    if len(input_list) % 2 != 0:
        input_list.append(["", ""])

    # Initialize the output list
    output_list = []

    # Iterate over the input list in steps of 2
    for i in range(0, len(input_list), 2):
        # Extract the current and next row
        row1 = input_list[i]
        row2 = input_list[i + 1]

        # Combine the two rows into a single row with 4 columns
        output_row = row1 + row2

        # Add the output row to the output list
        output_list.append(output_row)

    return output_list


def create_table_string(title, table_data, show_press=True):
    # Check if all rows have the same length
    row_lengths = [len(row) for row in table_data]
    if len(set(row_lengths)) > 1:
        raise ValueError("All rows must have the same length")

    # Split cells into lines
    table_data_split = []
    for row in table_data:
        row_split = []
        for cell in row:
            cell_lines = str(cell).splitlines()
            row_split.append(cell_lines)
        table_data_split.append(row_split)

    # Calculate the maximum number of lines per row
    max_lines_per_row = [max(len(cell) for cell in row) for row in table_data_split]

    # Calculate the maximum width of each column
    max_widths = [0] * len(table_data[0])
    for row in table_data_split:
        for j, cell in enumerate(row):
            for line in cell:
                max_widths[j] = max(max_widths[j], len(line))

    # Create the table string
    table_string = f"""
{title}
{'=' * len(title)}
"""
    # Create the top border
    top_border = Border.TOP_LEFT
    for width in max_widths:
        top_border += (
            Border.HORIZONTAL_DOUBLE * (width + 2) + Border.DOUBLE_TO_SINGLE_TOP
        )
    top_border = top_border[:-1] + Border.TOP_RIGHT
    table_string += top_border + "\n"

    # Create the header row
    header_row_lines = []
    for _ in range(max_lines_per_row[0]):
        header_row = Border.VERTICAL
        for i, cell in enumerate(table_data_split[0]):
            if _ < len(cell):
                header_row += " " + cell[_].ljust(max_widths[i]) + " " + Border.VERTICAL
            else:
                header_row += " " * (max_widths[i] + 2) + Border.VERTICAL
        header_row_lines.append(header_row)
    for line in header_row_lines:
        table_string += line + "\n"

    # Create the middle border
    middle_border = Border.DOUBLE_TO_SINGLE_LEFT
    for width in max_widths:
        middle_border += Border.HORIZONTAL_DOUBLE * (width + 2) + Border.CROSS
    middle_border = middle_border[:-1] + Border.DOUBLE_TO_SINGLE_RIGHT
    table_string += middle_border + "\n"

    # Create the data rows
    end = len(table_data_split[1:]) - 1
    for i, row in enumerate(table_data_split[1:]):
        row_lines = []
        for _ in range(max_lines_per_row[i + 1]):
            data_row = Border.VERTICAL
            for j, cell in enumerate(row):
                if _ < len(cell):
                    data_row += (
                        " " + cell[_].ljust(max_widths[j]) + " " + Border.VERTICAL
                    )
                else:
                    data_row += " " * (max_widths[j] + 2) + Border.VERTICAL
            row_lines.append(data_row)
        for line in row_lines:
            table_string += line + "\n"
        if i != end:
            table_string += middle_border + "\n"

    # Create the bottom border
    bottom_border = Border.BOTTOM_LEFT
    for width in max_widths:
        bottom_border += (
            Border.HORIZONTAL_DOUBLE * (width + 2) + Border.DOUBLE_TO_SINGLE_BOTTOM
        )
    bottom_border = bottom_border[:-1] + Border.BOTTOM_RIGHT
    table_string += bottom_border

    if show_press:
        table_string += "\n\nPress enter to continue\n-----------------------"

    return table_string


class KeyModifiers:
    CTRL = "CTRL"
    META = "META"
    RAW = "RAW"


META_TIMEOUT = 0.25  # Seconds
SYNTAX_TIMEOUT = 1100  # Milliseconds


def decode_utf8_key(first_byte, getch):
    """Passed the first byte of an expected utf-8 sequence will attempt to read
    the rest or retun -1 on failure."""
    # Determine the expected length of the UTF-8 sequence
    num_bytes = 0
    if 0xC0 <= first_byte <= 0xDF:
        num_bytes = 2  # 2-byte sequence
    elif 0xE0 <= first_byte <= 0xEF:
        num_bytes = 3  # 3-byte sequence
    elif 0xF0 <= first_byte <= 0xF7:
        num_bytes = 4  # 4-byte sequence
    else:
        LOGGER.inf("Invald UTF-8 start byte in decode_utf8_key: {first_byte}")
        return first_byte  # Invalid UTF-8 start byte, return as-is

    # Read the remaining bytes
    utf8_bytes = [first_byte]
    for _ in range(num_bytes - 1):
        next_byte = getch()
        if next_byte == -1 or next_byte & 0xC0 != 0x80:
            return -1  # Timeout or invalid UTF-8 continuation byte
        utf8_bytes.append(next_byte)

    try:
        return ord(bytes(utf8_bytes).decode("utf-8"))
    except UnicodeDecodeError as e:
        LOGGER.exception(e)
        return -1  # Return -1 on decoding failure


# TODO: key processing relies on the SYNTAX_TIMEOUT which is a mixing of concerns.
# we should set a global timeout for key handling and then use a different value for
# syntax.
def process_key(stdscr):
    CTRL, META, RAW = KeyModifiers.CTRL, KeyModifiers.META, KeyModifiers.RAW

    ch = stdscr.getch()
    if ch == -1:
        return -1
    meta = False
    if ch == KEY_ESCAPE:
        t0 = time.monotonic()
        ch = stdscr.getch()
        t1 = time.monotonic()
        if t1 - t0 < META_TIMEOUT:
            meta = True
        else:
            if ch != -1:
                curses.ungetch(ch)
            return (RAW, KEY_ESCAPE)

    # Unicode?
    # It is not clear how unicode and arrow keys will work together.
    if ch > 127:
        if ch > 255:
            return (RAW, ch)
        ch = decode_utf8_key(ch, stdscr.getch)
        if ch == -1:
            return -1

    # Exceptions that should be returned unmodified (even though they're control codes)
    raw_exceptions = {8, 9, 10, 27, 127}  # backspace, tab, newline, escape, delete

    if ch in raw_exceptions:
        return (RAW, ch)

    if meta:
        key_type = META
    else:
        key_type = CTRL if 1 <= ch <= 26 else RAW

    if key_type == CTRL or (meta and 1 <= ch <= 26):
        ch = ch + ord("a") - 1
    else:
        ch = ch

    return (key_type, ch)


class TextStore:
    def __init__(self):
        self._lines = {}
        self._reverse = {}
        self._lock = threading.Lock()

    def store(self, text):
        return BACK_GROUND_POOL.submit(self._process_text, text)

    def _process_text(self, text):
        lines = text.split("\n")
        line_indices = []

        for line in lines:
            if line in self._lines:
                index = self._lines[line]
            else:
                with self._lock:
                    if line in self._lines:
                        index = self._lines[line]
                    else:
                        index = len(self._lines)
                        self._lines[line] = index
                        self._reverse[index] = line
            line_indices.append(index)
        return line_indices

    def retrieve(self, future):
        line_indices = future.result()
        reversed_dict = self._reverse
        return "\n".join(reversed_dict[idx] for idx in line_indices)


class SimpleEditor:
    def __init__(self, stdscr, text="", lexer=None, filename=None):
        self.stdscr = stdscr
        self.text = text
        self.lexer = lexer
        self.undo_history = []
        self.redo_history = []
        self.cursor_x = 0
        self.cursor_y = 0
        self.scroll_y = 0
        self.split_text_into_lines()
        self.find_pattern = None
        self.selection = None
        self.selection_version = None
        self.find_pattern = None
        self.find_index = None
        self.find_version = None
        self.find_locations = None
        self.filename = filename
        self.version = 0
        self.text_store = TextStore()

        CTRL, META, RAW = KeyModifiers.CTRL, KeyModifiers.META, KeyModifiers.RAW

        def rni():
            raise ValueError("Not Implemented")

        self.yank_and_delete_selection = rni
        self.delete_selection = rni

        self.dispatch_table = {
            (CTRL, ord("a")): (self.start_of_line, "Goto start of line"),
            (CTRL, ord("e")): (self.end_of_line, "Goto end of line"),
            (CTRL, ord("f")): (self.page_down, "Page down"),
            (CTRL, ord("b")): (self.page_up, "Page up"),
            (CTRL, ord("u")): (self.undo, "Undo"),
            (CTRL, ord("r")): (self.redo, "Redo"),
            (CTRL, ord("x")): (lambda: True, "Save and exit"),
            (CTRL, ord("y")): (self.yank, "Yank lines"),
            (CTRL, ord("p")): (self.unyank, "Unyank lines"),
            (CTRL, ord("d")): (self.delete_line, "Delete line"),
            (META, ord("f")): (self.find_and_next, "Set find and goto next found"),
            (META, ord("F")): (self.find_and_next, "Set find"),
            (META, ord("n")): (
                self.find_next_from_cursor,
                "Goto next found from cursor",
            ),
            (META, ord("g")): (self.goto_line, "Goto line from 0"),
            (META, ord("G")): (self.goto_line1, "Goto line from 1"),
            (META, ord("i")): (self.info, "Show information"),
            (META, ord("h")): (self.help, "Show help"),
            (META, ord("y")): (self.yank_many, "Range yank"),
            (META, ord("m")): (self.yank_and_delete_many, "Range yank & delete"),
            (META, ord("p")): (self.unyank_before, "Unyank lines before cursor"),
            (META, ord("d")): (self.delete_many, "Range delete"),
            (META, ord("Y")): (self.yank_selection, "Selection yank"),
            (META, ord("P")): (self.unyank_selection, "Selection unyank"),
            (META, ord("M")): (
                self.yank_and_delete_selection,
                "Selection yank & delete - not implemented",
            ),
            (META, ord("D")): (
                self.delete_selection,
                "Selection delete - not implemented",
            ),
            (META, ord("s")): (self.save, "Save"),
            (META, ord("l")): (self.save, "Reload"),
            (META, ord(",")): (self.start_select, "Start select"),
            (META, ord(".")): (self.end_select, "End select"),
            (META, ord("/")): (self.deselect, "Deselect"),
            (META, ord(">")): (self.find_next, "Goto next found on index"),
            (META, ord("<")): (self.find_previous, "Goto previous found on index"),
            (META, ord("g")): (self.goto_line, "Goto line"),
            (META, ord("X")): (rni, "Execute an editing script"),
            (META, ord("q")): (self.escape, "Escape without saving"),
            (META, ord("r")): (self.replace, "Replace using a regex"),
            (RAW, curses.KEY_BACKSPACE): (self.delete_char, "Delete char"),
            (RAW, curses.KEY_DC): (self.delete_char, "Delete char"),
            (RAW, curses.KEY_DOWN): (self.cursor_down, "Cursor down"),
            (RAW, curses.KEY_LEFT): (self.cursor_left, "Cursor left"),
            (RAW, curses.KEY_RIGHT): (self.cursor_right, "Cursor right"),
            (RAW, curses.KEY_UP): (self.cursor_up, "Cursor up"),
            (RAW, curses.KEY_RESIZE): (self.resize, "Terminal resize event"),
            (RAW, KEY_ESCAPE): (self.escape, "Escape without saving (slow)"),
        }

    def set_lexer(self, lexer):
        self.lexer = lexer

    def edit(self):
        # Restore to that last edit entry not when the instance was created.
        self.original_text = self.text
        try:
            # Delayed highlight time out.
            self.stdscr.timeout(SYNTAX_TIMEOUT)
            self.render(True)
            previous_version = self.version
            while True:
                try:
                    key_info = process_key(self.stdscr)
                    if key_info == -1:
                        self.render(full=True)
                        previous_version = self.version
                    else:
                        action = self.dispatch_table.get(key_info)
                        if action is not None:
                            if action[0]() is True:
                                break
                        elif key_info[0] == KeyModifiers.RAW:
                            self.insert_char(chr(key_info[1]))
                    self.render(previous_version == self.version)
                    previous_version = self.version

                except KeyboardInterrupt:
                    if self.escape():
                        break
                except Exception as e:
                    handle_exception(self.stdscr, e)
                    try:
                        # Handle transient render issues.
                        self.render(False)
                    except Exception as e:
                        handle_exception(self.stdscr, e)
                        break
            if self.text != self.original_text:
                self.text = clean_text(self.text)
            return self.text
        finally:
            self.stdscr.timeout(-1)

    def escape(self):
        if handle_quit(self.stdscr):
            return
        self.text = self.original_text
        self.split_text_into_lines()
        return True

    def info(self):
        text_line = None
        cursor_y = self.cursor_y
        for idx, (start, end) in enumerate(self.index_screen_lines()):
            if start <= cursor_y and end >= cursor_y:
                text_line = idx

        info_data = [
            ("Lines", len(self.screen_lines)),
            ("Chars", len(self.text)),
            ("SLine", self.cursor_y),
            ("TLine", text_line),
            ("Colm", self.cursor_x),
        ]
        if self.filename:
            info_data.append(("File", self.filename))

        max_label_width = max(len(label) for label, _ in info_data)
        message = "\n".join(
            f"* {label.ljust(max_label_width)}: {value}" for label, value in info_data
        )

        show_command_panel(
            self.stdscr,
            f"""## Information
{message}""",
            True,
            win_height=len(info_data) + 12,
        )

    def resize(self):
        self.split_text_into_lines()
        self.selection = None
        self.scroll_y = 0
        self.cursor_y = 0
        self.cursor_x = 0
        self.version += 1

    def split_text_into_lines(self):
        lines = split_lines(self.text)
        screen_lines = []
        _, width = self.stdscr.getmaxyx()

        for line in lines:
            llen = len(line)
            start = 0
            if line == "\n":
                screen_lines[-1] += "\n"
            else:
                while llen - start > width:
                    screen_lines.append(line[start : start + width])
                    start += width
                screen_lines.append(line[start:])
                if len(screen_lines[-1].rstrip("\n")) == width:
                    screen_lines.append("")

        self.screen_lines = screen_lines

    def index_screen_lines(self):
        _, width = self.stdscr.getmaxyx()
        index = []
        screen_idx = 0
        for line in self.text.split("\n"):
            screen_start = screen_idx
            llen = len(line)
            start = 0
            while llen - start > width:
                screen_idx += 1
                start += width
            screen_idx += 1
            if len(self.screen_lines[-1].rstrip("\n")) == width:
                screen_idx += 1
            index.append([screen_start, screen_idx])
        return index

    def goto_line1(self):
        self.goto_line(one_based=True)

    def goto_line(self, one_based=False):
        if one_based:
            msg = "## Enter One Based Line Number:"
        else:
            msg = "## Enter Text Line Number:"
        number = show_command_panel(self.stdscr, msg)
        if number is ESCAPE:
            return
        line_index = self.index_screen_lines()
        for txt_at, (start, _) in enumerate(line_index):
            if start == self.cursor_y:
                break
        else:
            txt_at = 0
        if one_based:
            number = (
                parse_single_point_range(number, len(line_index) + 1, txt_at + 1) - 1
            )
        else:
            number = parse_single_point_range(number, len(line_index), txt_at)
        self.scan_to(line_index[number][0], 0)

    def regenerate_text(self):
        self.text = join_lines(self.screen_lines)

    def render(self, full):
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        # Work around blank line stripping in the lexer.
        offset = 0
        processed_text = []
        scroll_y = self.scroll_y
        if full:
            text = self.text
        else:
            processed_text = [(0, "\n") for _ in range(scroll_y)]
            slines = self.screen_lines
            slen = len(slines)
            text = "".join(slines[min(scroll_y, slen) : min(scroll_y + h, slen)])

        tend = len(text)
        while offset < tend and text[offset] == "\n":
            processed_text.append([(0, "\n")])
            offset += 1

        lexer = self.lexer
        if lexer == "ansiescape":
            lexer = "text"
        processed_text.extend(text_processor(text[offset:], w, lexer=lexer))
        y = 0
        stdscr = self.stdscr
        for line in processed_text:
            if y < scroll_y:
                y += 1
                continue
            if y >= scroll_y + h:
                break
            across = 0
            for attr, value in line:
                safe_addstr(stdscr, h, y - scroll_y, across, value, attr)
                across += len(value)
            y += 1

        y = max(0, self.cursor_y - scroll_y)
        y = min(y, h - 1)
        x = max(0, self.cursor_x)
        x = min(x, w - 1)
        self.paint_selection(y, x)
        self.stdscr.move(y, x)
        self.stdscr.refresh()
        curses.curs_set(1)

    def paint_selection(self, y, x):
        if self.selection is None:
            return

        if self.selection_version != self.version:
            self.selection_version = None
            self.selection = None
            return

        h, w = self.stdscr.getmaxyx()
        if len(self.selection) == 2:
            tly, tlx = self.selection
            bry, brx = y, x
        else:
            tly, tlx, bry, brx = self.selection
            bry = max(0, bry - self.scroll_y)
            bry = min(bry, h - 1)

        tly = max(0, tly - self.scroll_y)
        tly = min(tly, h - 1)
        tlx = max(0, tlx)
        tlx = min(tlx, w - 1)
        self.mark_background(
            (min(tly, bry), min(tlx, brx)), (max(tly, bry), max(tlx, brx))
        )

    def mark_background(self, top_left, bottom_right):
        for y in range(top_left[0], bottom_right[0] + 1):
            for x in range(top_left[1], bottom_right[1] + 1):
                char = self.stdscr.inch(y, x)
                if char & 0xFF == 0:
                    continue
                try:
                    self.stdscr.addch(
                        y,
                        x,
                        char & 0xFF,  # TODO: Fix this for unicode.
                        CursesDefs.BRIGHT_LILAC | curses.A_REVERSE,
                    )
                except curses.error as e:
                    LOGGER.info(f"Error in mark_background y, x = {y}, {x}")
                    LOGGER.exception(e)

    def current_line_length(self):
        return len(self.screen_lines[self.cursor_y].strip("\n"))

    def cursor_up(self):
        if self.cursor_y > 0:
            self.cursor_y -= 1
            if self.cursor_y < self.scroll_y:
                self.scroll_y -= 1

    def cursor_down(self):
        if self.cursor_y < len(self.screen_lines) - 1:
            self.cursor_y += 1
            if self.cursor_y >= self.scroll_y + self.stdscr.getmaxyx()[0]:
                self.scroll_y += 1

    def start_of_line(self):
        self.cursor_x = 0

    def end_of_line(self):
        self.cursor_x = self.current_line_length()
        if self.cursor_x >= self.stdscr.getmaxyx()[1] - 1:
            self.start_of_line()
            self.cursor_down()

    def page_down(self):
        h, _ = self.stdscr.getmaxyx()
        for _ in range(h):
            self.cursor_down()

    def page_up(self):
        h, _ = self.stdscr.getmaxyx()
        for _ in range(h):
            self.cursor_up()

    def cursor_left(self):
        if self.cursor_x > 0:
            self.cursor_x -= 1
        else:
            if self.cursor_y > 0:
                self.cursor_up()
                self.cursor_x = self.current_line_length()

    def cursor_right(self):
        if self.cursor_x < self.current_line_length() or (
            self.selection and len(self.selection) == 2
        ):
            self.cursor_x += 1
            if self.cursor_x >= self.stdscr.getmaxyx()[1]:
                if (not self.selection) or (len(self.selection) != 2):
                    self.start_of_line()
                    self.cursor_down()
                else:
                    self.cursor_x -= 1
        else:
            if self.cursor_y < len(self.screen_lines) - 1:
                self.cursor_down()
                self.start_of_line()

    def insert_char(self, char):
        if char == "\t":
            for _ in range(4):
                self.insert_char(" ")
            return

        self.append_undo()
        screen_lines = self.screen_lines
        line = screen_lines[self.cursor_y]
        self.cursor_x = min(len(line.strip("\n")), self.cursor_x)
        if self.cursor_x == 0:
            line = char + line
        else:
            line = line[: self.cursor_x] + char + line[self.cursor_x :]
        screen_lines[self.cursor_y] = line
        self.cursor_x += 1
        self.regenerate_text()
        self.split_text_into_lines()
        if self.cursor_x >= self.stdscr.getmaxyx()[1] or char == "\n":
            self.start_of_line()
            self.cursor_down()

    def regen_and_check_y(self):
        self.regenerate_text()
        self.split_text_into_lines()
        if self.cursor_y >= len(self.screen_lines):
            self.cursor_y = len(self.screen_lines) - 1

    def delete_char(self):
        self.append_undo()
        screen_lines = self.screen_lines
        line = screen_lines[self.cursor_y]
        self.cursor_x = min(len(line.rstrip("\n")), self.cursor_x)
        if self.cursor_x == 0:
            if self.cursor_y == 0:
                return
            self.cursor_y -= 1
            line = screen_lines[self.cursor_y]
            line = line[:-1]
            screen_lines[self.cursor_y] = line + screen_lines[self.cursor_y + 1]
            del screen_lines[self.cursor_y + 1]
            self.cursor_x = len(line)
        else:
            self.screen_lines[self.cursor_y] = line = (
                line[: self.cursor_x - 1] + line[self.cursor_x :]
            )
            self.cursor_x -= 1

        self.regen_and_check_y()

    def delete_line(self):
        self.append_undo()
        del self.screen_lines[self.cursor_y]
        self.regen_and_check_y()

    def deselect(self):
        self.selection = None
        self.render(False)

    def start_select(self):
        self.selection = [self.cursor_y, self.cursor_x]
        self.selection_version = self.version
        self.render(False)

    def end_select(self):
        self.selection.extend([self.cursor_y, self.cursor_x])
        if self.selection[0:2] == self.selection[2:4]:
            self.selection = None
        self.render(False)

    def check_selection(self):
        if (self.selection is None) or (len(self.selection)) != 4:
            show_command_panel("**No selection found**", True)
            return False
        else:
            return True

    def yank_selection(self):
        if not self.check_selection():
            return
        tly, tlx, bry, brx = self.selection
        tly, tlx, bry, brx = (
            min(tly, bry),
            min(tlx, brx),
            max(tly, bry),
            max(tlx, brx),
        )

        yanked = []
        for y in range(tly, bry + 1):
            current_len = len(self.screen_lines[y])
            if tlx >= current_len:
                continue
            current_right = min(current_len, brx + 1)
            if tlx < current_right:
                yanked.append(self.screen_lines[y][tlx:current_right].rstrip("\n"))
        Status.yanked.set_current([Prompt(role=Role.USER, text="\n".join(yanked))])

    def delete_many(self):
        range_def = show_command_panel(self.stdscr, "## Enter range to delete?", False)
        if range_def is ESCAPE:
            return
        self.delete_lines(self.parse_index_range(range_def))

    def delete_lines(self, index_range):
        self.append_undo()
        for index in sorted(set(index_range), reverse=True):
            del self.screen_lines[index]
        self.regen_and_check_y()

    def yank(self):
        Status.yanked.set_current(
            [Prompt(role=Role.USER, text=self.screen_lines[self.cursor_y])]
        )

    def parse_index_range(self, range_def):
        return parse_index_range(range_def, len(self.screen_lines), self.cursor_y)

    def yank_many(self):
        range_def = show_command_panel(self.stdscr, "## Enter range to yank?", False)
        if range_def is ESCAPE:
            return
        self.yank_lines(self.parse_index_range(range_def))

    def yank_lines(self, index_range):
        yanked = [self.screen_lines[line] for line in index_range]
        Status.yanked.set_current([Prompt(role=Role.USER, text=join_lines(yanked))])

    def yank_and_delete_many(self):
        range_def = show_command_panel(
            self.stdscr, "## Enter range to yank and delete?", False
        )
        if range_def is ESCAPE:
            return
        index_range = self.parse_index_range(range_def)
        self.yank_lines(index_range)
        self.delete_lines(index_range)

    def unyank_before(self, offset=1):
        self.unyank(0)

    def unyank(self, offset=1):
        yanked = Status.yanked.get_current()
        if not check_yank(self.stdscr, yanked):
            return
        yanked = split_lines(
            "\n".join(prompt.text for prompt in Status.yanked.get_current())
        )
        self.append_undo()
        self.screen_lines = (
            self.screen_lines[: self.cursor_y + offset]
            + yanked
            + self.screen_lines[self.cursor_y + offset :]
        )
        self.regenerate_text()
        self.split_text_into_lines()

    def unyank_selection(self):
        yanked = Status.yanked.get_current()
        if not check_yank(self.stdscr, yanked):
            return
        yanked = split_lines(
            "\n".join(prompt.text for prompt in Status.yanked.get_current())
        )
        self.append_undo()
        y = self.cursor_y
        x = self.cursor_x
        lines = self.screen_lines
        for yanked_line in yanked:
            if y >= len(lines):
                lines.append("")
            line = self.screen_lines[y]
            has_nl = line.endswith("\n")
            line = line.rstrip("\n")
            if x > len(line):
                line += " " * (x - len(line))
            line = line[:x] + yanked_line + line[x:]
            if has_nl:
                line = line + "\n"
            self.screen_lines[y] = line
            y += 1
        self.regenerate_text()
        self.split_text_into_lines()

    def create_undo_entry(self):
        return (
            self.cursor_x,
            self.cursor_y,
            self.scroll_y,
            self.text_store.store(self.text),
        )

    def append_undo(self):
        self.version += 1
        self.undo_history.append(self.create_undo_entry())
        self.redo_history.clear()

    def set_from_history(self, hst_tuple):
        self.cursor_x, self.cursor_y, self.scroll_y, index = hst_tuple
        self.text = self.text_store.retrieve(index)
        self.split_text_into_lines()
        self.version += 1

    def undo(self):
        if self.undo_history:
            self.redo_history.append(self.create_undo_entry())
            self.set_from_history(self.undo_history.pop())

    def redo(self):
        if self.redo_history:
            self.undo_history.append(self.create_undo_entry())
            self.set_from_history(self.redo_history.pop())

    def help(self):
        title = "Editor Controls"
        table_data = [
            ["Key Binding", "Action"],
        ]

        # Generate table data from dispatch table
        for key, (_, description) in self.dispatch_table.items():
            key_binding = ""
            if key[0] == KeyModifiers.CTRL:
                key_binding += f"Ctrl + {chr(key[1])}"
            elif key[0] == KeyModifiers.META:
                key_binding += f"Meta + {chr(key[1])}"
            elif key[0] == KeyModifiers.RAW:
                if key[1] == curses.KEY_BACKSPACE:
                    key_binding += "Backspace"
                elif key[1] == curses.KEY_DC:
                    key_binding += "Delete"
                elif key[1] == curses.KEY_DOWN:
                    key_binding += "Down Arrow"
                elif key[1] == curses.KEY_LEFT:
                    key_binding += "Left Arrow"
                elif key[1] == curses.KEY_RIGHT:
                    key_binding += "Right Arrow"
                elif key[1] == curses.KEY_UP:
                    key_binding += "Up Arrow"
                elif key[1] == KEY_ESCAPE:
                    key_binding += "Escape"

            table_data.append([key_binding, description])

        with curses_step_out(self.stdscr):
            print_markdown_highlighted(
                create_table_string(title, double_column_list(table_data))
            )
            input()

    def find_all_matches(self):
        regex = re.compile(self.find_pattern)
        matches = []
        lines = self.screen_lines

        i = 0
        while i < len(lines):
            search_str = lines[i]
            end_idx = i
            # List of (line index, start offset in concatenated string)
            line_offsets = [(i, 0)]

            # Continue concatenating until a line ends with '\n' or we reach the end
            while end_idx + 1 < len(lines) and not lines[end_idx].endswith("\n"):
                end_idx += 1
                # Store offset for future lines
                line_offsets.append((end_idx, len(search_str)))
                search_str += lines[end_idx]

            # Search for all matches in the concatenated string
            for match in regex.finditer(search_str):
                match_start = match.start()

                # Find which line contains the match start
                # Default to the first line in this concatenation
                correct_list_index = i
                for line_idx, offset in line_offsets:
                    if match_start >= offset:
                        correct_list_index = line_idx  # Update to the correct line
                    else:
                        break  # We've found the correct line

                # Compute the offset in that specific line
                start_offset_in_line = (
                    match_start - line_offsets[correct_list_index - i][1]
                )
                matches.append((correct_list_index, start_offset_in_line))

            # Move to the next independent segment
            i = end_idx + 1

        # Return all matches as a list of (index, offset) tuples
        return matches

    def set_find(self):
        find_pattern = show_command_panel(
            self.stdscr,
            f"""# Regex to find:
> [{self.find_pattern if self.find_pattern else ''}]""",
        )
        if find_pattern is ESCAPE:
            return
        if find_pattern == "":
            if self.find_pattern == "":
                show_command_panel(self.stdscr, "**Igoring Empty Pattern**", True)
                return
            else:
                find_pattern = self.find_pattern

        self.find_pattern = find_pattern
        self.find_version = self.version
        self.find_locations = self.find_all_matches()
        self.find_index = -1

    def find_and_next(self):
        self.set_find()
        self.find_next_from_cursor()

    def find_next_from_cursor(self):
        if not self.find_check():
            return
        if self.cursor_y > self.find_locations[-1][0]:
            y, x = self.find_locations[0]
            self.scan_to(y, x)
            self.find_index = 0
            return
        self.find_index = 0
        # Find correct line
        current_location = (self.cursor_y, self.cursor_x)
        while self.find_locations[self.find_index] <= current_location:
            self.find_index += 1
            # Have we gone off the end.
            if self.find_index >= len(self.find_locations):
                self.find_next()
                return
        self.scan_to(*self.find_locations[self.find_index])

    def find_check(self):
        if self.find_pattern is None:
            self.set_find()
        if self.find_version != self.version:
            self.find_locations = self.find_all_matches()
            if self.find_index >= len(self.find_locations):
                self.find_index = 0
            self.find_version = self.version
        if not self.find_locations:
            show_command_panel(
                self.stdscr,
                f"**Not Found**\n* {self.find_pattern}",
                True,
            )
            return False
        return True

    def find_next(self):
        if not self.find_check():
            return
        self.find_index += 1
        if self.find_index >= len(self.find_locations):
            self.find_index = 0
        y, x = self.find_locations[self.find_index]
        self.scan_to(y, x)

    def find_previous(self):
        if not self.find_check():
            return
        self.find_index -= 1
        if self.find_index < 0:
            self.find_index = len(self.find_locations) - 1
        y, x = self.find_locations[self.find_index]
        self.scan_to(y, x)

    def scan_to(self, y, x):
        self.cursor_x = x
        start = time.monotonic()
        while self.cursor_y > y:
            for _ in range(max(abs(self.cursor_y - y) // 500, 1)):
                self.cursor_up()
            if time.monotonic() - start < SCROLLY_OPTS.max_scroll_time:
                self.render(True)
        while self.cursor_y < y:
            for _ in range(max(abs(self.cursor_y - y) // 500, 1)):
                self.cursor_down()
            if time.monotonic() - start < SCROLLY_OPTS.max_scroll_time:
                self.render(True)
        self.render(True)

    def screen_lines_to_text_lines(self, screen_lines):
        scr_text = {}
        for idx, (start, end) in enumerate(self.index_screen_lines()):
            for scr_line in range(start, end + 1):
                scr_text[scr_line] = idx
        text_lines = set()
        for idx in screen_lines:
            text_lines.add(scr_text[idx])
        return text_lines

    def replace(self):
        replace_pattern = show_command_panel(
            self.stdscr,
            """# range/pattern/replace</g>:
* `range` is a standard range expression.
* `pattern` is a regex.
* `replace` is what to replace with.
* `g` is the optional global modifier""",
            win_height=20,
            win_width=45,
        )
        if replace_pattern is ESCAPE:
            return
        parts = parse_vim_sub(replace_pattern)
        len_parts = len(parts)
        if len_parts < 3 or len_parts > 4:
            raise ValueError(f"Failed to parse {replace_pattern}")
        parts.append(len_parts == 3 or parts[3] != "g")
        range_list = self.parse_index_range(parts[0])
        self.regex_replace(range_list, parts[1], parts[2], parts[3])

    def regex_replace(self, range_list, match, replace, global_replace):
        text_lines = self.screen_lines_to_text_lines(range_list)
        lines_list = self.text.split("\n")
        for line_number in text_lines:
            if not 0 <= line_number < len(lines_list):
                raise IndexError(
                    f"Line number {line_number} is out of bounds. Valid line numbers are 1 to {len(lines_list)}"
                )
        for line_number in text_lines:
            line = lines_list[line_number]
            if global_replace:
                new_line = re.sub(match, replace, line)
            else:
                new_line = re.sub(match, replace, line, count=1)
            lines_list[line_number] = new_line
        modified_text = "\n".join(lines_list)
        self.append_undo()
        self.text = modified_text
        self.split_text_into_lines()

    def check_is_file(self):
        if self.filename is None:
            show_command_panel(self.stdscr, "**Not a file!**", True)
            return False
        return True

    def save(self):
        if not self.check_is_file():
            return

        with utf8_open(self.filename, "r") as f:
            file_text = f.read()

        if file_text != self.original_text:
            reload = ask_if_reload(self.stdscr)
            if reload == ReloadOptions.RUN_AWAY:
                show_command_panel(self.stdscr, "**File untouched**", True)
                return
            if reload == ReloadOptions.RELOAD:
                self.text = file_text
                self.original_text = self.text
                self.regen_and_check_y()
                show_command_panel(self.stdscr, "# Reloaded\n* Skipping save.", True)
                return

        modified_text = clean_text(self.text)
        if modified_text != self.text:
            self.text = modified_text
            self.split_text_into_lines()

        if self.original_text == self.text:
            show_command_panel(self.stdscr, "# No changes\n* Skipping save.", True)
        else:
            with open(self.filename, "w") as f:
                f.write(self.text)
            self.original_text = self.text
            show_command_panel(self.stdscr, "# File saved OK.", True)

    def load(self):
        with utf8_open(self.filename, "r") as f:
            file_text = f.read()
        self.text = file_text
        self.regen_and_check_y()


class SCLLexer(RegexLexer):
    name = "SCL"
    aliases = ["scl"]
    filenames = ["*.scl"]

    tokens = {
        "root": [
            (r"^\s*#.*", lx_token.Comment),
            (r"\s+", lx_token.Text),
            (r"^\s*[a-zA-Z0-9_]+:", lx_token.Keyword, "token"),
            (r'"', lx_token.String.Double),
            (r"'", lx_token.String.Single),
            (r"\d+", lx_token.Number),
            (r'[^\s"#\']+', lx_token.Name.Attribute, "token"),
        ],
        "token": [
            (r"\\", lx_token.Text, "escape"),  # Transition to escape state
            (r'[^\s"#\'\\]+', lx_token.Text),  # Match any remaining text
        ],
        "escape": [
            (
                r'([abfvrnt"\'\\])',
                lx_token.String.Escape,
                "token",
            ),  # Match simple escape sequences
            (
                r"([0-7]{1,3})",
                lx_token.String.Escape,
                "token",
            ),  # Match octal escape sequences
            (
                r"x([0-9a-fA-F]{1,2})",
                lx_token.String.Escape,
                "token",
            ),  # Match hexadecimal escape sequences
            (
                r"u([0-9a-fA-F]{4})",
                lx_token.String.Escape,
                "token",
            ),  # Match unicode escape sequences
            (
                r"U([0-9a-fA-F]{8})",
                lx_token.String.Escape,
                "token",
            ),  # Match unicode escape sequences
            (r".", lx_token.Text, "token"),  # Match any other character
        ],
    }


class SCLAutomator:
    def command_show_range(self, index_range):
        """Print out the interpretation of a range."""
        prompts, _ = Status.conversations.get_current_conversation()
        proc_index_range = parse_index_range(index_range, len(prompts), self.current)
        proc_index_range = ", ".join(str(index) for index in proc_index_range)
        self.command_print(
            f"* The range: `{index_range}`\n* Expands to: `{proc_index_range}`"
        )

    def command_print(self, message):
        """Print to a regular console in markdown."""
        with curses_step_out(self.stdscr):
            print_markdown_highlighted(message)
            input("Press enter to continue")

    def command_new(self, role, syntax, text):
        """Create a none file new prompt at the end of the current conversation."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        role = role.upper()
        if not Role.is_valid(role):
            raise ValueError(f"Role not valid `{role}`")
        get_lexer_by_name(syntax)
        if role == Role.FILE:
            raise ValueError("Use 'new_file:' to create file prompts")
            filename = text
            with utf8_open(filename, "r") as file:
                text = file.read()
        do_new(prompts, role, text, syntax=syntax)
        if role == Role.FILE:
            prompts[-1].filename = filename
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )
        background_summarise_prompt(prompts[-1])

    def command_new_file(self, syntax, filename, text):
        """Create a new file prompt at the end of the current conversation."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        get_lexer_by_name(syntax)
        do_new(prompts, Role.FILE, text, syntax=syntax)
        prompts[-1].filename = filename
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )
        background_summarise_prompt(prompts[-1])

    def command_syntax(self, syntax, index_range):
        """Set prompts' syntax highlighting language."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        get_lexer_by_name(syntax)
        index_range = parse_index_range(index_range, len(prompts), self.current)
        for index in index_range:
            prompts[index].syntax = syntax
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_slot(self, slot):
        """Set the currently active slot."""
        slot = int(slot)
        Status.conversations.set_current_index(slot)

    def command_bash(self, index_range, stdin):
        """Run a range of prompts if bash passing in stdin if it is not empty."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        index_range = parse_index_range(index_range, len(prompts), self.current)
        for index in index_range:
            prompt = prompts[index]
            if prompt.role != Role.BASH:
                raise ValueError(f"Prompt role {prompt.role} is not {Role.BASH}")
            text = execute_bash_script(prompt.text, stdin if stdin else None)
            prompts.append(Prompt(role=Role.USER, text=text))
            Status.conversations.set_current_conversation(
                Conversation(prompts, scroll_offset)
            )
            background_summarise_prompt(prompts[-1])

    def command_bash_unyank(self, index_range):
        """Run an index range of prompts of bash passing in the current yank as stdin."""
        yanked = Status.yanked.get_current()
        if not check_yank(self.stdscr, yanked):
            raise ValueError("No current yank")
        yanked = "\n".join(prompt.text for prompt in Status.yanked.get_current())
        self.command_bash(index_range, yanked)

    def command_execute(self, index_range):
        """Passes each conversation in the range to the AI and records the return."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        index_range = parse_index_range(index_range, len(prompts), self.current)
        original_prompts = prompts[:]
        for index in index_range:
            local_prompts = original_prompts[: index + 1]
            handle_execute(self.stdscr, local_prompts)
            prompts.append(local_prompts[-1])
            background_summarise_prompt(prompts[-1])
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def check_are_file(self, index_range):
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        index_range = parse_index_range(index_range, len(prompts), self.current)
        for index in index_range:
            if prompts[index].role != Role.FILE:
                raise ValueError(
                    f"Roll not file for file operation {prompts[index].role}"
                )
        return prompts, scroll_offset, index_range

    def command_write_file(self, index_range):
        """Causes the FILE role prompts in range to write to out their contents."""
        prompts, scroll_offset, index_range = self.check_are_file(index_range)
        for index in index_range:
            prompt = prompts[index]
            with utf8_open(prompt.filename, "w") as file:
                file.write(prompt.text)

    def command_read_file(self, index_range):
        """Causes the FILE role prompts in range to read their contents discarding current."""
        prompts, scroll_offset, index_range = self.check_are_file(index_range)
        for index in index_range:
            prompt = prompts[index]
            with utf8_open(prompt.filename, "r") as file:
                prompt.text = file.read()
            background_summarise_prompt(prompt)
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_role(self, role, index_range):
        """Set the role for the prompts in the range."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        role = role.upper()
        if not Role.is_valid(role):
            raise ValueError(f"Role {role} is not valid")
        index_range = parse_index_range(index_range, len(prompts), self.current)
        for index in index_range:
            prompts[index].role = role
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_delete(self, index_range):
        """Delete the prompts in the range to undelete buffer."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        index_range = sorted(
            set(parse_index_range(index_range, len(prompts), self.current))
        )
        for idx in index_range:
            Status.conversations.add_deleted(prompts[idx])
        for idx in reversed(index_range):
            del prompts[idx]
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_undelete(self, index_range):
        """Undelete the prompts in the range from undelete buffer."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        for prompt in Status.conversations.undelete(index_range):
            prompts.append(prompt)
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_yank(self, index_range):
        """Place copies of the range of prompts into the yank buffer."""
        prompts, _ = Status.conversations.get_current_conversation()
        index_range = parse_index_range(index_range, len(prompts), self.current)
        Status.yanked.set_current([prompts[index] for index in index_range])

    def command_unyank_index(self, index_range):
        """Iterate over the yank buffer placing its prompts in the range of locations clearing the buffer as we go."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        if len(prompts) == 0:
            indexes = parse_index_range(index_range, 1, self.current)
            if indexes != [0]:
                raise IndexError(f"Index range not valid {index_range}")
        else:
            indexes = parse_index_range(index_range, len(prompts), self.current)
        rev_yanks = list(reversed(Status.yanked.get_current()))
        for index in reversed(indexes):
            if len(rev_yanks) == 0:
                raise IndexError(
                    f"Exhausted yanks before exhausting index_range {index_range}"
                )
            yank = rev_yanks.pop().copy()
            background_summarise_prompt(prompts[-1])
            prompts = prompts[:index] + [yank] + prompts[index:]
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_unyank(self):
        """Unyanks to the end of the current slot."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        yanked = Status.yanked.get_current()
        if yanked is not None:
            prompts.extend(y.copy() for y in yanked)
            background_summarise_prompt(prompts[-1])
        else:
            raise ValueError("Nothing to unyank")
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_automate(self, index_range):
        """Automate each SCL prompt in index_range; potentially recusively."""
        # We make no effort to protect the system from the fact it will reindex the prompts as the automation
        # continues. therefore, within the limits of the system we can automated what we have generated from an
        # automate which makes this recurse.
        # Say we automate 1,2. 1 creates 2 and 3. now 2 automates 3,4 and 3 makes 4 and 5 and so on we now will
        # keep going till the Python vm gives up in some way.
        prompts, _ = Status.conversations.get_current_conversation()
        index_range = parse_index_range(index_range, len(prompts), self.current)
        for index in index_range:
            prompts, scroll_offset = Status.conversations.get_current_conversation()
            automator = SCLAutomator(self.stdscr, prompts, scroll_offset)
            automator.execute_statements(prompts[index].text)

    def command_merge(self, index_range):
        """Place the merge of all the prompts in index_range as a USER prompt in the yank buffer."""
        prompts, scroll_offset = Status.conversations.get_current_conversation()
        index_range = parse_index_range(index_range, len(prompts), self.current)
        texts = []
        for index in index_range:
            texts.append(prompts[index].text)
        prompts.append(Prompt(role=Role.USER, text="\n".join(texts)))
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def command_set_current(self, index_range):
        """Move the current marker `.` to the passed single point range."""
        prompts, _ = Status.conversations.get_current_conversation()
        index = parse_single_point_range(index_range, len(prompts), self.current)
        self.current = index

    def command_help(self):
        """Create a prompt containing help for SCL."""
        docs = self.document()
        self.command_new(
            "COMMENT", "markdown", create_table_string("SCL Help:", docs, False)
        )

    def command_scl(self, version):
        """Control scl settings."""

    # SCL Machinery below

    def __init__(self, stdscr, prompts, scroll_offset):
        self.stdscr = stdscr
        self.current = None
        Status.conversations.set_current_conversation(
            Conversation(prompts, scroll_offset)
        )

    def execute_command(self, name, *args):
        # Convert name to lowercase for case-insensitive matching
        name = name.lower()

        # Find the matching command method
        method_name = f"command_{name}"
        method = getattr(self, method_name, None)

        if method is None:
            raise AttributeError(f"Command '{name}' not found")

        # Get the method signature
        num_args_required = method.__code__.co_argcount - 1
        # Check if the number of arguments matches
        if len(args) != num_args_required:
            raise TypeError(
                f"Command '{name}' requires {num_args_required} arguments, but {len(args)} were provided"
            )

        # Call the method with the provided arguments
        method(*args)

    @classmethod
    def document(cls):
        documentation = []
        documentation = [("Command", "Parameters", "Description")]
        for method_name in dir(cls):
            if method_name.startswith("command_"):
                method = getattr(cls, method_name)
                name = method_name[len("command_") :]
                args = " ".join(tuple(inspect.signature(method).parameters.keys())[1:])
                docstring = method.__doc__.strip()
                docstring = "\n".join(textwrap.wrap(docstring, 40))
                documentation.append((name, args, docstring))
        return documentation

    def parse_statements(self, input_raw):
        lines = self.preprocess_input(input_raw)
        statements = []

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line or stripped_line.startswith("#"):
                continue

            match = re.match(r"^([a-zA-Z0-9_]+):", line)
            if match:
                command = match.group(1)
                buffer = line[match.end() :].strip()
                parsed_data = self.parse_buffer(buffer)
                statements.append((command,) + tuple(parsed_data))
            else:
                raise ValueError(f"Invalid input format: {line[:20]}")

        return statements

    def preprocess_input(self, input_str):
        if not input_str:
            return []

        buffer = bytearray()
        quote = {ord("'"), ord('"')}
        quote_byte = None
        back_slash = 0
        for b in input_str.encode("utf-8"):
            if b == ord("\\"):
                back_slash += 1
                buffer.append(b)
                continue
            elif b not in quote:
                # End of backslash sequence and we don't
                # need to retain knowledge of the state of backslashes
                back_slash = 0

            if b in quote:
                # Quote was not backslashed
                if back_slash % 2 == 0:
                    # Leave matching quote
                    if b == quote_byte:
                        quote_byte = None
                    # Enter quote
                    elif quote_byte is None:
                        quote_byte = b
                # Append quote
                buffer.append(b)
                # Clear backslash state
                back_slash = 0
                continue

            # Clear backslash state
            back_slash = 0
            if quote_byte is None:
                buffer.append(b)
                continue

            if b == ord("\n"):
                buffer.extend(b"\\n")
                continue

            buffer.append(b)

        return buffer.decode("utf-8").split("\n")

    def parse_buffer(self, buffer):
        if buffer is None or buffer == "":
            return ()

        utf8_buffer = buffer.encode("utf-8")

        tokens = []
        token = bytearray()
        in_quote = False
        quote_char = None
        consecutive_backslashes = 0
        space = (32, 9, 10, 13)
        quote = (34, 39)

        for byte in utf8_buffer:
            if consecutive_backslashes > 0:
                token.append(byte)
                consecutive_backslashes -= 1
                continue

            if in_quote:
                if byte == 92:
                    consecutive_backslashes = 1
                    token.append(byte)
                elif byte == quote_char:
                    in_quote = False
                    quote_char = None
                    tokens.append(bytes(token))
                    token.clear()
                else:
                    token.append(byte)
            else:
                if byte in quote:
                    in_quote = True
                    quote_char = byte
                    if token:
                        tokens.append(bytes(token))
                        token.clear()
                elif byte in space:
                    if token:
                        tokens.append(bytes(token))
                        token.clear()
                else:
                    token.append(byte)

        if token:
            tokens.append(bytes(token))

        tokens = [token.decode("utf-8") for token in tokens]

        escapes = {
            '"': '"',
            "'": "'",
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "b": "\b",
            "f": "\f",
            "v": "\x0b",
            "\\": "\\",
        }

        def unicode_escape(match):
            if match.group(1) in escapes:
                return escapes[match.group(1)]
            elif match.group(2) is not None:
                # Handle \uXXXX
                code_point = match.group(2)
                return chr(int(code_point, 16))
            elif match.group(3) is not None:
                # Handle \UXXXXXXXX
                code_point = match.group(3)
                return chr(int(code_point, 16))
            elif match.group(4) is not None:
                # Handle \xXX
                code_point = match.group(4)
                return chr(int(code_point, 16))
            else:
                return match.group(0)

        regex = re.compile(
            r"\\(['\"ntrbfv\\])|\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|\\x([0-9a-fA-F]{2})"
        )
        return [regex.sub(unicode_escape, token) for token in tokens]

    def execute_statements(self, input_raw):
        statements = self.parse_statements(input_raw)
        for statement in statements:
            command = statement[0].lower()
            args = statement[1:]
            self.execute_command(command, *args)


class SCLRecorder:
    ESCAPE_RE = re.compile(r"([\\\"\x00-\x09\x0B-\x1F])")
    SCL_HEADER = "scl: 1.0"

    @classmethod
    def serialize_token(cls, raw_string):
        # Split the string into segments; escapable characters are captured.
        parts = cls.ESCAPE_RE.split(raw_string)

        fragments = ['"']  # start with an opening quote

        for part in parts:
            if not part:
                continue  # skip empty segments
            # If the part matches the escapable pattern, process it.
            if cls.ESCAPE_RE.fullmatch(part):
                ch = part
                if ch == "\\":
                    fragments.append("\\\\")
                elif ch == '"':
                    fragments.append('\\"')
                elif ord(ch) < 32:
                    fragments.append("\\x{:02X}".format(ord(ch)))
            else:
                # Otherwise, it's a safe fragment—append as is.
                fragments.append(part)

        fragments.append('"')  # close the string with a quote
        return "".join(fragments)

    @classmethod
    def conversation_to_scl(cls, prompts, header=False):
        if header:
            scl = [cls.SCL_HEADER]
        else:
            scl = []
        scl.append("delete: -")
        for prompt in prompts:
            syntax = prompt.syntax or "markdown"
            text = cls.serialize_token(prompt.text)
            role = prompt.role
            if role == Role.FILE:
                filename = prompt.filename
                scl.append(f"new_file: {syntax} {filename} {text}")
            else:
                scl.append(f"new: {role} {syntax} {text}")
        return scl

    @classmethod
    def conversations_to_scl(cls, header=False):
        if header:
            scl = [cls.SCL_HEADER]
        else:
            scl = []
        for index, convo in enumerate(Status.conversations):
            scl.append(f"slot: {index}")
            scl.extend(cls.conversation_to_scl(convo.prompts))
        scl.append("slot: 0")
        return scl

    @classmethod
    def try_load(cls, stdscr, text):
        if not text.startswith("scl:"):
            return False
        automator = SCLAutomator(stdscr, [], 0)
        automator.execute_statements(text)
        return True


@contextlib.contextmanager
def utf8_open(filename, mode):
    if "b" in mode:
        raise ValueError("Binary mode is not allowed. Use text mode only.")
    file = open(filename, mode, encoding="utf-8", errors="backslashreplace")
    try:
        yield file
    finally:
        file.close()


def check_yank(stdscr, yanked):
    if yanked is None:
        show_command_panel(stdscr, "**Nothing to unyank!**")
        return False
    return True


def parse_vim_sub(to_parse):
    parts = []
    current = []
    i = 0
    n = len(to_parse)
    while i < n:
        c = to_parse[i]
        if c == "/":
            j = i - 1
            bs_count = 0
            while j >= 0 and to_parse[j] == "\\":
                bs_count += 1
                j -= 1
            if bs_count % 2 == 0:
                parts.append("".join(current))
                current = []
                i += 1
                continue
        current.append(c)
        i += 1

    if len(parts) > 2:
        parts[2] = parts[2].replace("\\/", "/")
    else:
        raise ValueError("Failed to parse the pattern; did you miss of the closing /?")

    parts.append("".join(current))
    return parts


def browse_for_file(stdscr, getter):
    file_browser = getter(stdscr)
    filename = file_browser.run()
    return file_browser.current_dir, filename


def handle_browse_load(stdscr, prompts):
    new_dir, filename = browse_for_file(stdscr, FileBrowsers.load_browser)
    if filename is not None:
        do_load(
            stdscr,
            relativise_filename(SCROLLY_OPTS.prompts_dir, new_dir, filename),
            prompts,
        )


def handle_browse_read(stdscr, prompts):
    new_dir, filename = browse_for_file(stdscr, FileBrowsers.load_browser)
    if filename is not None:
        do_read(
            stdscr,
            relativise_filename(SCROLLY_OPTS.prompts_dir, new_dir, filename),
            prompts,
        )


def handle_browse_save(stdscr, prompts, scroll_offset):
    new_dir, filename = browse_for_file(stdscr, FileBrowsers.save_browser)
    if filename is not None:
        do_save(
            relativise_filename(SCROLLY_OPTS.prompts_dir, new_dir, filename),
            prompts,
            scroll_offset,
        )


def handle_browse_write(stdscr, prompts, scroll_offset):
    new_dir, filename = browse_for_file(stdscr, FileBrowsers.save_browser)
    if filename is not None:
        do_save(
            relativise_filename(SCROLLY_OPTS.prompts_dir, new_dir, filename),
            prompts,
            scroll_offset,
            True,
        )


def do_file(filename, prompts):
    absname = create_file_path(SCROLLY_OPTS.prompts_dir, filename)
    with open(absname, "r") as read_file:
        text = read_file.read()
    prompts.append(Prompt(role=Role.FILE, text=text, filename=absname))
    background_summarise_prompt(prompts[-1])


def handle_file(stdscr, prompts):
    filename = show_command_panel(stdscr, ">Enter filename to read:")
    if filename is ESCAPE:
        return
    do_file(filename, prompts)


def handle_browse_file(stdscr, prompts):
    new_dir, filename = browse_for_file(stdscr, FileBrowsers.file_browser)
    if filename is not None:
        do_file(
            relativise_filename(SCROLLY_OPTS.prompts_dir, new_dir, filename), prompts
        )


def handle_toggle_summary():
    global SUMMARY_VIEW
    if SUMMARY_VIEW == ViewModes.SUMMARY_ALL:
        SUMMARY_VIEW = ViewModes.VIEW_LAST
    elif SUMMARY_VIEW == ViewModes.VIEW_LAST:
        SUMMARY_VIEW = ViewModes.VIEW_ALL
    else:
        SUMMARY_VIEW = ViewModes.SUMMARY_ALL


COMMAND_PREFIX = "command_"


def pretty_print_log_to_markdown(log):
    markdown = ""
    for i, entry in enumerate(log):
        timestamp = datetime.datetime.fromtimestamp(entry["time"]).astimezone()
        markdown += f"### Log `{entry['type']}` Entry {i+1}\n"
        markdown += f"* Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z')}\n"
        markdown += f"* Text: {entry['text']}\n"
        if "stack_trace" in entry:
            markdown += "* Stack Trace:\n"
            for j, frame in enumerate(entry["stack_trace"]):
                markdown += f"  + Frame {j+1}:\n"
                markdown += f"    - File: {frame['filename']}\n"
                markdown += f"    - Line: {frame['lineno']}\n"
                markdown += f"    - Function: {frame['name']}\n"
                markdown += f"    - Code: {frame['line']}\n"
        markdown += "\n"
    return markdown


def command_show_log(stdscr, *args):
    with curses_step_out(stdscr):
        print_markdown_highlighted(pretty_print_log_to_markdown(LOGGER.get_log()))
        input("Press enter to continue")


def get_commands():
    commands = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and name.startswith(COMMAND_PREFIX):
            commands[name] = obj
    return commands


def command_be_off_the_wall(*args):
    SCROLLY_OPTS.temperature = 1.5
    SCROLLY_OPTS.top_p = 1.0


def command_be_inventive(*args):
    SCROLLY_OPTS.temperature = 1.2
    SCROLLY_OPTS.top_p = 0.95


def command_be_normal(*args):
    SCROLLY_OPTS.temperature = 0.8
    SCROLLY_OPTS.top_p = 0.9


def command_be_boring(*args):
    SCROLLY_OPTS.temperature = 0.5
    SCROLLY_OPTS.top_p = 0.8


def command_return(*args):
    pass


def command_file_manager(stdscr, *args):
    browse_for_file(stdscr, FileBrowsers.command_browser)


def command_merge_current_slot(stdscr, prompts, scroll_offset):
    text = "\n".join(p.text for p in prompts)
    prompts.clear()
    prompts.append(Prompt(role=Role.COMMENT, text=text))


def command_clear_all_slots(stdscr, prompts, scroll_offset):
    prompts.clear()
    scroll_offset = 0
    for index in range(len(Status.conversations)):
        Status.conversations.set_conversation(
            index, Conversation(prompts=[], scroll_offset=0)
        )
    auto_save(prompts, scroll_offset)


def command_show_status(stdscr, prompts, scroll_offset):
    Status.conversations.set_current_conversation(
        Conversation(prompts=prompts, scroll_offset=scroll_offset)
    )
    with curses_step_out(stdscr):
        print("Scrolly Status")
        print("==============")
        print(f"- Last loaded:   {Status.last_loaded}")
        print(f"- Last saved:    {Status.last_saved}")
        print(f"- Last written:  {Status.last_written}")
        print(f"- Last read:     {Status.last_read}")
        print(f"- Has yank:      {Status.yanked.get_current() is not None}")
        print(f"- Model:         {SCROLLY_OPTS.model}")
        print(f"- Temperature:   {SCROLLY_OPTS.temperature}")
        print(f"- Top_p:         {SCROLLY_OPTS.top_p}")
        print()
        print("Slots:")
        for index, conversation in enumerate(Status.conversations):
            found = False
            for prompt in conversation.prompts:
                if prompt.role == Role.TITLE:
                    print(f"- Slot {index:>2}: {prompt.text}")
                    found = True
                    break
            if not found:
                print(f"- Slot {index:>2}: No Title")
        input("Press enter to continue")


def command_style_chooser(stdscr, *args):
    def get_style():
        with curses_step_out(stdscr):
            print("Available Styles:")
            print("==================")
            styles = sorted(get_all_styles())
            max_cols = 3
            max_width = 80
            col_width = (
                max_width - 4
            ) // max_cols  # subtract 4 for padding and borders
            for i, style in enumerate(styles):
                index = i + 1
                print(f"{index:>2}) {style:<{col_width}}", end="")
                if (i + 1) % max_cols == 0 or i == len(styles) - 1:
                    print()
                else:
                    print(" | ", end="")
            print()
            return styles[int(input("Choose style number: ")) - 1]

    while True:
        try:
            SCROLLY_OPTS.style = get_style()
            text_processor.cache_clear()
            return
        except (ValueError, IndexError) as e:
            LOGGER.exception(e)
            pass


def command_set_model(stdscr, *args):
    with curses_step_out(stdscr):
        print(f"Current Model: {SCROLLY_OPTS.model}")
        model = input(f"New Model (enter={DEFAULT_MODEL}):")
        if model.strip() == "":
            model = DEFAULT_MODEL
        SCROLLY_OPTS.model = model


def command_shell(stdscr, *args):
    with curses_step_out(stdscr):
        subprocess.run([SCROLLY_OPTS.shell])


def handle_command(stdscr, prompts, scroll_offset):
    def get_command():
        with curses_step_out(stdscr):
            print("Availible Commands:")
            print("===================")
            cmds = []
            for index, v in enumerate(sorted(get_commands().items())):
                name, func = v
                print(f"  {index:>2})\t{name[len(COMMAND_PREFIX):].replace('_', ' ')}")
                cmds.append(func)
            print()
            cmd = input("Choose cmd number: ")
            LOGGER.info(f"Running command {cmd}")
            return cmds[int(cmd)]

    while True:
        try:
            return get_command()(stdscr, prompts, scroll_offset)
        except (ValueError, IndexError) as e:
            LOGGER.exception(e)
            pass


def relativise_filename(current_dir, new_dir, filename):
    """
    Updates the file name to include the relative path from current_dir to new_dir
    """
    current_dir_path = pathlib.Path(current_dir)
    new_dir_path = pathlib.Path(new_dir)

    if current_dir_path == new_dir_path:
        return filename
    else:
        relative_path = os.path.relpath(new_dir_path, start=current_dir_path)
        updated_filename = os.path.join(relative_path, filename)
        return updated_filename


# Main function
def curses_main(stdscr):
    CursesDefs.setup()
    curses.meta(True)
    prompts = []

    if SCROLLY_OPTS.read is not None:
        do_read(stdscr, SCROLLY_OPTS.read, prompts)

    dispatch_table = {
        # Mutators.
        ord("e"): lambda: handle_edit(stdscr, prompts),
        ord("d"): lambda: handle_delete(stdscr, prompts),
        ord("n"): lambda: handle_new(stdscr, prompts),
        ord("o"): lambda: handle_role(stdscr, prompts),
        ord("f"): lambda: handle_file(stdscr, prompts),
        ord("F"): lambda: handle_browse_file(stdscr, prompts),
        ord("t"): lambda: handle_automate(stdscr, prompts, scroll_offset),
        # ord("T"): lambda: handle_reautomate(stdscr, prompts),
        ord("l"): lambda: handle_load(stdscr, prompts),
        ord("L"): lambda: handle_browse_load(stdscr, prompts),
        ord("r"): lambda: handle_read(stdscr, prompts),
        ord("R"): lambda: handle_browse_read(stdscr, prompts),
        ord("j"): lambda: handle_paste(stdscr, prompts),
        ord("x"): lambda: handle_execute(stdscr, prompts),
        ord("k"): lambda: handle_yank_to_bash(stdscr, prompts),
        ord("K"): lambda: handle_reyank_to_bash(stdscr, prompts),
        ord("X"): lambda: handle_bash(stdscr, prompts),
        ord("Z"): lambda: handle_rerun_bash(stdscr, prompts),
        ord("u"): lambda: handle_undelete(stdscr, prompts),
        ord("m"): lambda: handle_move(stdscr, prompts),
        ord("p"): lambda: handle_unyank(stdscr, prompts),
        ord("^"): lambda: handle_syntax(stdscr, prompts),
        ord("!"): lambda: handle_command(stdscr, prompts, scroll_offset),
        ord("|"): lambda: handle_read_from_stdin(stdscr, prompts),
        # No change.
        ord("i"): lambda: handle_info(stdscr),
        ord("q"): lambda: handle_quit(stdscr),
        ord("v"): lambda: handle_view(stdscr, prompts),
        ord("c"): lambda: handle_copy(stdscr, prompts),
        ord("C"): lambda: handle_copy_all(stdscr, prompts),
        ord("h"): lambda: handle_help_system(stdscr),
        ord("s"): lambda: handle_save(stdscr, prompts, scroll_offset),
        ord("S"): lambda: handle_browse_save(stdscr, prompts, scroll_offset),
        ord("w"): lambda: handle_write(stdscr, prompts, scroll_offset),
        ord("W"): lambda: handle_browse_write(stdscr, prompts, scroll_offset),
        ord("y"): lambda: handle_yank(stdscr, prompts),
        ord(">"): lambda: handle_scroll_to_bottom(stdscr, prompts, scroll_offset),
        ord("<"): lambda: handle_scroll_to_top(stdscr, prompts, scroll_offset),
        ord("0"): lambda: handle_slot(prompts, scroll_offset, 0),
        ord("1"): lambda: handle_slot(prompts, scroll_offset, 1),
        ord("2"): lambda: handle_slot(prompts, scroll_offset, 2),
        ord("3"): lambda: handle_slot(prompts, scroll_offset, 3),
        ord("4"): lambda: handle_slot(prompts, scroll_offset, 4),
        ord("5"): lambda: handle_slot(prompts, scroll_offset, 5),
        ord("6"): lambda: handle_slot(prompts, scroll_offset, 6),
        ord("7"): lambda: handle_slot(prompts, scroll_offset, 7),
        ord("8"): lambda: handle_slot(prompts, scroll_offset, 8),
        ord("9"): lambda: handle_slot(prompts, scroll_offset, 9),
        ord("_"): lambda: handle_toggle_summary(),
        curses.KEY_RIGHT: lambda: handle_up(scroll_offset),
        curses.KEY_LEFT: lambda: handle_down(stdscr, scroll_offset),
        curses.KEY_DOWN: lambda: handle_up_page(stdscr, prompts, scroll_offset),
        curses.KEY_UP: lambda: handle_down_page(stdscr, prompts, scroll_offset),
    }

    mutators = "deEfFjkKlLmnNoprRtTuxXZm!|^"

    scroll_offset = 0

    def main_run_loop():
        nonlocal scroll_offset
        while True:
            handle_scroll(stdscr, prompts, scroll_offset)
            c = stdscr.getch()
            try:
                result = dispatch_table.get(c, lambda: None)()
            finally:
                # Control which handlers should trigger an auto-save.
                if chr(c) in mutators:
                    auto_save(prompts, scroll_offset)

            if result is False:
                stdscr.clear()
                stdscr.refresh()
                break
            elif isinstance(result, int):
                scroll_offset = result

    while True:
        try:
            main_run_loop()
            return
        except KeyboardInterrupt:
            if not handle_quit(stdscr):
                break
        except Exception as e:
            handle_exception(stdscr, e)


@datafields
class ScrollyOpts:
    __fields__ = (
        "read",
        "key",
        "model",
        "auto_save",
        "prompts_dir",
        "max_tokens",
        "top_p",
        "temperature",
        "shell",
        "style",
        "style_overrides",
        "fun",
        "fake_bold",
        "max_scroll_time",
        "summary_cache_size",
    )


def parse_cmdline():
    def parse_dict(arg):
        result = {}
        pairs = arg.split(",")
        for pair in pairs:
            if ":" not in pair:
                raise argparse.ArgumentTypeError(
                    "Invalid key-value pair: {}".format(pair)
                )
            key, value = pair.split(":")
            result[key.strip()] = value.strip()
        return result

    parser = argparse.ArgumentParser(description="Conversation AI Script")
    parser.add_argument("--read", help="Read a project file.")
    parser.add_argument("--key", required=True, help="Set the metagen key to use")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Set the AI model to use"
    )
    parser.add_argument(
        "--auto-save", help="Automatically save the conversation to the specified file"
    )
    parser.add_argument(
        "--prompts-dir",
        default=str(pathlib.Path.home() / "scrolly-prompts"),
        help="Set the directory for scrolly prompts load/append/save",
    )
    parser.add_argument(
        "--max-tokens", type=int, default="4096", help="Max tokens to use per execution"
    )
    parser.add_argument("--top-p", type=float, default="0.9", help="model top_p value")
    parser.add_argument(
        "--temperature", type=float, default="0.8", help="model tempurature value"
    )
    parser.add_argument(
        "--shell",
        default=os.environ.get("SHELL", "/bin/sh"),
        help="shell to use",
    )
    parser.add_argument(
        "--style",
        default="default",
        help="pygments color pallet",
    )
    parser.add_argument(
        "--style-overrides",
        type=parse_dict,
        default={},
        help="override pygments style with key-value pairs (e.g., 'markdown:friendly,python:emacs')",
    )
    parser.add_argument(
        "--fun",
        action="store_true",
        help="Have more fun",
    )
    parser.add_argument(
        "--fake-bold",
        action="store_true",
        help="Fake bold if needed [windows]",
    )
    parser.add_argument(
        "--max-scroll-time",
        default="3.0",
        type=float,
        help="Maximum time to allow scrolling in scans and paging",
    )
    parser.add_argument(
        "--summary-cache-size",
        default="16384",
        type=int,
        help="Number of LRU records held in the persistent prompt summary cache",
    )

    args = parser.parse_args()
    LOGGER.info(f"Starting Scrolly with args {args}")

    validate_directory(args.prompts_dir)

    return ScrollyOpts(
        read=args.read,
        key=args.key,
        model=args.model,
        auto_save=args.auto_save,
        prompts_dir=args.prompts_dir,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        shell=args.shell,
        style=args.style,
        style_overrides=args.style_overrides,
        fun=args.fun,
        fake_bold=args.fake_bold,
        max_scroll_time=args.max_scroll_time,
        summary_cache_size=args.summary_cache_size,
    )


SCROLLY_OPTS = None


INTRO_CMAP = {
    "P": [
        "#########",
        "# #####  #",
        "#        #",
        "#        #",
        "# #######",
        "##",
        "##",
        "##",
        "##",
        "# ",
    ],
    "A": [
        "  #######",
        " # ##### #",
        "#         #",
        "#         #",
        "# ####### #",
        "###########",
        "##       ##",
        "##       ##",
        "##       ##",
        "#        #",
    ],
    "C": [
        " #########",
        "######### #",
        "#",
        "##",
        "##",
        "##",
        "##",
        "#",
        "########  #",
        " #########",
    ],
    "E": [
        " ##########",
        "##########",
        "#",
        "#",
        "########",
        "#######",
        "#",
        "#",
        "#########",
        " ##########",
    ],
    "I": [
        " ##########",
        "##########",
        "    #",
        "   ##",
        "   ##",
        "   ##",
        "   ##",
        "   # ",
        "#########",
        " ##########",
    ],
    "-": [
        "",
        "",
        "",
        "",
        "  #######",
        " #######",
    ],
}


def draw_pace_character(stdscr, x, y, char, attr):
    """Draws a character at the specified position"""
    for i, line in enumerate(INTRO_CMAP[char]):
        for j, pixel in enumerate(line):
            if pixel == "#":
                try:
                    stdscr.addstr(y + i, x + j, char, attr)
                    stdscr.refresh()
                    c = stdscr.getch()
                    if c != -1:
                        curses.ungetch(c)
                        break
                except curses.error:
                    pass


def draw_pace(stdscr):
    char_width = 10
    try:
        if not SCROLLY_OPTS.fun:
            stdscr.timeout(0)
            word = "PACE-IIE"
            x = 2
            y = 2
            for i, char in enumerate(word):
                draw_pace_character(
                    stdscr, x + i * (char_width + 2), y, char, CursesDefs.WHITE
                )
            stdscr.timeout(-1)
            c = stdscr.getch()
            curses.ungetch(c)
            return

        stdscr.timeout(10)
        cols = [
            CursesDefs.BRIGHT_MAGENTA,
            CursesDefs.BRIGHT_YELLOW,
            CursesDefs.BRIGHT_RED,
            CursesDefs.BRIGHT_CYAN,
            CursesDefs.BRIGHT_GREEN,
            CursesDefs.BRIGHT_BLUE,
            CursesDefs.BRIGHT_WHITE,
        ]
        word = "IIE"
        while True:
            if word == "IIE":
                stdscr.erase()
            height, width = stdscr.getmaxyx()
            if word == "PACE":
                word = "IIE"
            else:
                word = "PACE"
            x = randint(0, width - 12)
            y = randint(0, height - 1)
            try:
                stdscr.addstr(y, x, r"¯\_(ツ)_/¯", cols[0])
            except curses.error:
                pass
            x = randint(-20, width)
            y = randint(-20, height)

            for i, char in enumerate(word):
                draw_pace_character(
                    stdscr, x + i * (char_width + 2), y, char, cols[i % len(cols)]
                )
            c = stdscr.getch()
            if c != -1:
                curses.ungetch(c)
                break
            cols = cols[1:] + cols[:1]
    finally:
        stdscr.timeout(-1)


def invoke_main():
    global SCROLLY_OPTS
    SCROLLY_OPTS = parse_cmdline()
    term = os.environ["TERM"]
    if term not in ("xterm-256color", "xterm-256color-italic"):
        input(
            "Terminal is not xterm-256color.\nSome features may fail\nPress enter to continue"
        )
        os.environ["TERM"] = "xterm-256color"
    curses.wrapper(curses_main)
    kill_executor_pools()


HELP_PREFIX = "help_"


def print_markdown_highlighted(text):
    """
    Print the Markdown highlighted version of the given text to stdout.

    :param text: The Markdown text to highlight
    """
    print_highlighted(text, "markdown")


def print_highlighted(text, lexer_name):
    if lexer_name == "ansiescape":
        lexer_name = "text"
    lexer = get_lexer_by_name(lexer_name)
    formatter = Terminal256Formatter(style=get_formatter_style(lexer_name))
    highlighted_text = highlight(text, lexer, formatter)
    sys.stdout.write(highlighted_text)


def reset_terminal():
    sys.stdout.write("\x1b[0m")  # Reset color
    sys.stdout.write("\x1b[H\x1b[2J")  # Clear screen and move cursor to top-left
    sys.stdout.flush()


def get_help_subjects():
    subjects = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and name.startswith(HELP_PREFIX):
            subjects[name] = obj
    return subjects


def handle_help_system(stdscr):
    def get_help_subject():
        msg = []
        msg.append("# Availible Subjects:")
        subjects = []
        for index, v in enumerate(sorted(get_help_subjects().items())):
            name, func = v
            msg.append(f"* {index:>2})\t{name[len(HELP_PREFIX):].replace('_', ' ')}")
            subjects.append(func)
        print_markdown_highlighted("\n".join(msg))
        print()
        return subjects[int(input("Choose cmd number: "))]

    with curses_step_out(stdscr):
        while True:
            try:
                hs_text = get_help_subject()()
                reset_terminal()
                print_markdown_highlighted(hs_text)
                print("\n")
                input("Press enter to continue:")
                reset_terminal()
                return
            except (ValueError, IndexError) as e:
                LOGGER.exception(e)
                pass


def help_key_strokes():
    msg = """
# Help:
*  d: delete    - delete a prompt
*  e: edit     - edit a prompt, simple
*  i: info      - show slot index etc
*  m: move      - move the order of a prompt
*  j: paste     - download paste as a new prompt
*  k: y-exec    - as per X inputting current yank
*  K: y-re-exec - as per Z but inputting current yank
*  n: new       - create a new prompt, simple
*  p: unyank    - append prompts from yank buffer
*  q: quit      - exit Scrolly
*  o: role      - change the role of a prompt
*  t: automate  - run last prompt as SCL
*  T: re-auto   - as per Z but scl
*  v: view      - view in plain text a prompt body
*  x: execute   - execute the current prompts set
*  X: bash      - run last prompt as bash
*  Z: re-bash   - move last bash prompt to front and (re)run
*  y: yank      - put one or more prompts in the yank buffer
*  ^: syntax    - choose syntax highlighting
*  !: command   - enter command mode
*  |: cat-in    - read from stdin to a prompt up to Ctrl^D
*  _: summary   - toggle summary view
*  h: help      - enter help mode

*  cC: copy     - create a paste of a prompt/conversation
*  fF: read     - read a file, name/browes
*  lL: load     - load a conversation, name/browse
*  rR: load     - read a project, name/browse
*  sS: save     - save the conversation, name/browse
*  wW: read     - write the project, name/browse
*  0-9: slot    - choose current conversation slot

* \u2190\u2192 scroll, \u2191\u2193 fast scroll, <> top/bottom
""".strip()
    return msg


def help_terminal_instructions():
    msg = """
# Terminals Are Flaky
Different terminals work in different ways and they are a balance between the server and the client.
To makes things more complex we have intermediaries like tmux and screen.
Helpfully, most of the time we can force the system to think it is xterm-256color.

## Bold On Windows
Bold does not seem to work for Windows clients. Use the --fake-bold flag for Scrolly and it will move
bold text towards pure white which works ok'ish.

## Setup to allow italics on Mac and Linux
If you find that highlighted shows italic as reversed then try following the
instructions on setting up your terminfo.

### Create definition
Create a file named xterm-256color-italic.terminfo:
```text
# A xterm-256color based TERMINFO that adds the escape sequences for italic.
xterm-256color-italic|xterm with 256 colors and italic,
  sitm=\\E[3m, ritm=\\E[23m,
  use=xterm-256color,
```

### Create the binary term info
```shell
tic xterm-256color-italic.terminfo
```

### To Use It
export TERM=xterm-256color-italic
""".strip()
    return msg


def help_index_range():
    msg = """
# Index-Range Micro Language
Index-range micro language, which allows users to specify a range of indices using a concise syntax. The language is designed to parse a comma-delimited string into a list of integer indices. It turns out this is a 'super power' for Scrolly. As mentioned elsewhere, Scrolly tightly integrates a text editor with a conversation editor. Specifying which prompts in a conversation to manipulate is pretty much the same thing as specifying which lines in a text file to manipulate.

When it came to SCL, the core language was made very much simpler than it would have been without index-range. By using index-range formulae as arguments to SCL commands the commands take on much greater power declaratively and thereby avoid the need for variables and loops.

## Syntax
The index-range micro language supports the following syntax elements:

* **Single integer**: A single integer value, e.g., `1`.
* **Range**: A range of integers, specified using a dash (`-`) separator, e.g., `1-3`.
* **Open-ended range**: A range that starts or ends at an unspecified index, e.g., `1-` or `-3`.
* **Relative ranges**: Ranges that are relative to a current index, specified using a dot (`.`) or a dollar sign (`$`), e.g., `.-3` or `$-3`.
* **Relative offsets**: Offsets from a current index, specified using a plus sign (`+`) or a tilde (`~`), e.g., `+3` or `~3`.
* **Current index**: A special symbol (`.`) that represents the current index.
* **Last index**: A special symbol (`$`) that represents the last valid index.

## Formal Grammar
The formal grammar for the index-range micro language can be defined as follows:
```markdown
# Index-Range Micro Language Grammar

 Terminals
* `INT`: a non-negative integer value
* `DOT`: a dot (`.`) symbol
* `DOLLAR`: a dollar sign (`$`) symbol
* `PLUS`: a plus sign (`+`) symbol
* `TILDE`: a tilde (`~`) symbol
* `DASH`: a dash (`-`) symbol
* `COMMA`: a comma (`,`) symbol

 Non-Terminals
* `EXPR`: an expression
* `RANGE`: a range
* `OFFSET`: an offset
* `RELATIVE`: a relative expression

 Productions
1. `EXPR` ::= `INT` | `RANGE` | `RELATIVE`
2. `RANGE` ::= `INT` `DASH` `INT` | `INT` `DASH` | `DASH` `INT`
3. `RELATIVE` ::= `DOT` `OFFSET` | `DOLLAR` `OFFSET`
4. `OFFSET` ::= `PLUS` `INT` | `TILDE` `INT`
5. `EXPR_LIST` ::= `EXPR` (`COMMA` `EXPR`)*

 Notes
* The `EXPR_LIST` production represents a comma-delimited list of expressions.
* The `RELATIVE` production allows for relative expressions using the dot (`.`) or dollar sign (`$`) symbols.
* The `OFFSET` production allows for offsets using the plus sign (`+`) or tilde (`~`) symbols.
```
## Semantics
The semantics of the index-range micro language are defined by the Python code, which parses the input string and generates a list of integer indices. The code handles errors and edge cases, such as invalid input syntax, out-of-range indices, and empty segments.

## Example Use Cases
The index-range micro language can be used in various applications, such as:

* Specifying a range of pages to print or display
* Selecting a subset of data from a larger dataset
* Defining a range of indices for a data structure or array

For example, the input string `"1-3, 5, 7-"` might be parsed into the list of indices `[1, 2, 3, 5, 7, 8, 9, ...]`, depending on the maximum valid index.### Index-Range Micro Language Examples
Here are 10 examples that demonstrate various aspects of the index-range micro language:

1. **Simple range**: `1-3`
    * Parses to: `[1, 2, 3]`
2. **Open-ended range**: `5-`
    * Parses to: `[5, 6, 7, ..., max_count - 1]`
3. **Relative range**: `.-2`
    * Parses to: `[current - 2, current - 1, current]` (assuming `current` is set)
4. **Relative offset**: `+3`
    * Parses to: `[current + 3]` (assuming `current` is set)
5. **Last index**: `$`
    * Parses to: `[max_count - 1]`
6. **Single index**: `7`
    * Parses to: `[7]`
7. **Multiple ranges**: `1-3, 5, 7-`
    * Parses to: `[1, 2, 3, 5, 7, 8, 9, ..., max_count - 1]`
8. **Relative range with offset**: `~2-`
    * Parses to: `[current - 2, current - 1, current, current + 1, ..., max_count - 1]` (assuming `current` is set)
9. **Empty range**: `-`
    * Parses to: `[0, 1, 2, ..., max_count - 1]` (all indices)
10. **Complex example**: `1-3, 5, $-2, .+1`
    * Parses to: `[1, 2, 3, 5, max_count - 1, max_count - 2, current + 1]` (assuming `current` is set)

### Note:
* These examples assume a maximum valid index (`max_count`) and a current index (`current`) are set, unless otherwise specified.
* The actual parsing results may vary depending on the specific implementation and input values.
* If `max_count` is zero the only valid range is `-` which resolves to [].
""".strip()
    return msg
