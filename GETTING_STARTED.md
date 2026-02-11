# Getting Started with Kutsu

Kutsu is a Python library for building composable, stateful HTTP request workflows. It lets you chain actions together using intuitive operators, pass state between them, and use a rich expression system for dynamic data transformation.

## Installation

```bash
# Using pip
pip install kutsu

# Using Poetry
poetry add kutsu

# With AWS authentication support
pip install kutsu[aws]
```

Requires Python 3.10+.

## Core Concepts

Kutsu has three building blocks: **State**, **Actions**, and **Expressions**.

### State

`State` is a key-value container that flows through your action pipeline. Create it with keyword arguments, dicts, or by subclassing.

```python
from kutsu import State

# From keyword arguments
s = State(name="alice", count=3)
print(s.name)   # "alice"
print(s["count"])  # 3

# From a dict
s = State({"host": "example.com", "port": 443})

# Subclass for reusable defaults
class MyConfig(State):
    host = "localhost"
    port = 8080

s = MyConfig(port=9090)  # host="localhost", port=9090

# Combine states (right side wins)
a = State(x=1, y=2)
b = State(y=3, z=4)
c = a + b  # x=1, y=3, z=4
```

### Actions

Actions are transformers that take a `State` and return a new `State`. There are synchronous (`Action`) and asynchronous (`AsyncAction`) variants.

```python
from kutsu import State, Action, AsyncAction

# Wrap a function
def add_greeting(state):
    return State(state, greeting=f"Hello, {state.name}!")

s = State(name="world") | add_greeting
print(s.greeting)  # "Hello, world!"

# Subclass Action for reusable logic
class Multiply(Action):
    def __call__(self, state=None):
        state = super().__call__(state)
        state.result = state.x * state.y
        return state

s = State(x=3, y=7) | Multiply()
print(s.result)  # 21

# Async actions
class FetchData(AsyncAction):
    async def call_async(self, state=None):
        state = await super().call_async(state)
        # ... do async work ...
        state.data = "fetched"
        return state

s = State() | FetchData()
```

### Piping and Chaining

Use `|` to pipe state into an action, and `>>` to chain actions together.

```python
from kutsu import State, Action, Override, Default, Slice, Identity

# Pipe state into an action
result = State(x=1) | my_action

# Pipe a dict into an action
result = {"x": 1} | my_action

# Chain actions with >>
chain = action_a >> action_b >> action_c
result = State() | chain

# Chain with inline functions
result = State(x=10) | (lambda s: State(s, x=s.x + 1)) >> (lambda s: State(s, x=s.x * 2))

# Built-in actions
result = State(a=1, b=2, c=3) | Slice("a", "c")  # keeps only a and c
result = State(a=1) | Default(a=0, b=2)            # a=1, b=2 (defaults don't override)
result = State(a=1) | Override(a=99)                # a=99 (override always wins)
```

### Parallel Execution

Run multiple async actions concurrently using `//` or `**`.

```python
from kutsu import AsyncAction, Parallel, State

class SlowTask(AsyncAction):
    async def call_async(self, state=None):
        state = await super().call_async(state)
        import asyncio
        await asyncio.sleep(0.1)
        state.done = True
        return state

# Parallelize two actions
par = SlowTask() // SlowTask()

# Run 10 instances in parallel
par = SlowTask() ** 10

# Execute and collect results
s = State() | par
for result_state in s.results:
    print(result_state.done)  # True
```

Each parallel action receives an `INSTANCE_NUMBER` in its state.

## Expressions

Expressions let you define dynamic values that are evaluated at runtime against the current state. They support string templates, variable references, arithmetic, comparisons, conditionals, and more.

### Variables and Templates

```python
from kutsu import State, Var, Env, Eval, evaluate

state = State(host="example.com", port=443)

# Var: reference a state variable
evaluate(Var("host"), state)  # "example.com"

# String templates using ${...} syntax
evaluate("https://${host}:${port}/api", state)  # "https://example.com:443/api"

# Env: read environment variables
evaluate(Env("HOME"), state)  # "/home/user"

# Eval action: resolve all expressions in a state
s = State(host="example.com", url="https://${host}/api")
s = s | Eval
print(s.url)  # "https://example.com/api"
```

### Arithmetic

Expressions support standard math operators, and they compose naturally with `Var`:

```python
from kutsu import Var, evaluate, State, Add, Mul

state = State(x=10, y=3)

evaluate(Var("x") + Var("y"), state)   # 13
evaluate(Var("x") - Var("y"), state)   # 7
evaluate(Var("x") * Var("y"), state)   # 30
evaluate(Var("x") / Var("y"), state)   # 3.333...
evaluate(Var("x") // Var("y"), state)  # 3
evaluate(Var("x") % Var("y"), state)   # 1
evaluate(Var("x") ** Var("y"), state)  # 1000

# Mix with literals
evaluate(Var("x") + 5, state)   # 15
evaluate(2 * Var("y"), state)   # 6
```

### Comparisons and Logic

```python
from kutsu import Var, evaluate, State, Eq, Gt, And, Or, Not, If, In

state = State(score=85, name="alice", active=True)

# Comparisons
evaluate(Var("score") > 70, state)       # True
evaluate(Var("score") == 100, state)     # False
evaluate(Var("name") == "alice", state)  # True

# Logical operators (use & and | on expressions)
evaluate(Var("active") & (Var("score") > 50), state)  # True
evaluate(Var("active") | (Var("score") < 50), state)  # True

# Conditional
evaluate(If("active", "yes", "no"), state)                       # "yes"
evaluate(If(Gt(Var("score"), 90), "excellent", "good"), state)   # "good"

# Membership
evaluate(In(Var("name"), ["alice", "bob"]), state)  # True
```

### Secrets

Mark sensitive values with `Secret` to mask them in logs and output:

```python
from kutsu import State, Var, Secret, evaluate

state = State(api_key=Secret("sk-12345"))

# Normal evaluation reveals the value
evaluate(Var("api_key"), state)                        # "sk-12345"

# Masked evaluation hides it
evaluate(Var("api_key"), state, mask_secrets=True)     # Masked[str]
evaluate("Key: ${api_key}", state, mask_secrets=True)  # "Key: ****MASKED****"
```

### Conditional Deletion

Use `Del` and `DelIfNone` to conditionally remove keys from dicts or items from lists:

```python
from kutsu import State, Var, Del, DelIfNone, evaluate

state = State(include_debug=None, user="admin")

# Del removes an entry
evaluate({"a": 1, "b": Del()}, state)  # {"a": 1}

# DelIfNone removes only if the value is None
evaluate({
    "user": Var("user"),
    "debug": DelIfNone(Var("include_debug")),
}, state)
# {"user": "admin"}  — debug key is removed because include_debug is None

# Shorthand: ~Var("x") is equivalent to DelIfNone(Var("x"))
evaluate({"debug": ~Var("include_debug")}, state)  # {}
```

### Select (Switch)

Map a value to different outputs:

```python
from kutsu import State, Select, evaluate

state = State(level="warn")

evaluate(Select("level", {
    "info": "All good",
    "warn": "Watch out",
    "error": "Something broke",
}, "Unknown level"), state)
# "Watch out"
```

## HTTP Requests

`HttpRequest` is the main async action for making HTTP calls. It supports dynamic URLs, headers, auth, JSON bodies — all driven by expressions and state.

```python
from kutsu import State, Var, Secret, HttpRequest

# Simple GET
class GetUser(HttpRequest):
    method = "GET"
    url = "https://api.example.com/users/${user_id}"

s = State(user_id="42") | GetUser()
# Prints request/response info, returns state with cookies etc.

# POST with JSON body
class CreateUser(HttpRequest):
    method = "POST"
    url = "https://api.example.com/users"
    json = {
        "name": Var("name"),
        "email": Var("email"),
    }

s = State(name="Alice", email="alice@example.com") | CreateUser()

# Process the response
class GetTodo(HttpRequest):
    url = "https://jsonplaceholder.typicode.com/todos/${todo_id}"
    quiet = True  # suppress output

    def on_response(self, state, response):
        state.todo = response.json()
        return state

s = State(todo_id="1") | GetTodo()
print(s.todo["title"])
```

### Authentication

```python
from kutsu import HttpRequest, Var, Secret

# Bearer token
class AuthenticatedRequest(HttpRequest):
    url = "https://api.example.com/data"
    auth_token = Secret(Var("token"))

# Basic auth
class BasicAuthRequest(HttpRequest):
    url = "https://api.example.com/data"
    auth_username = Var("username")
    auth_password = Secret(Var("password"))

# AWS Sigv4 (requires kutsu[aws])
from kutsu.http_request import AwsSigV4Auth

class AWSRequest(HttpRequest):
    url = "https://my-api.execute-api.us-east-1.amazonaws.com/prod/data"
    auth = AwsSigV4Auth(
        service="execute-api",
        region="us-east-1",
        access_key=Var("aws_access_key"),
        secret_key=Secret(Var("aws_secret_key")),
    )
```

### Display Options

Control what gets printed during requests:

```python
class VerboseRequest(HttpRequest):
    url = "https://example.com"
    verbose = True  # show everything: headers, state, SSL info, etc.

class QuietRequest(HttpRequest):
    url = "https://example.com"
    quiet = True  # suppress all output

class CustomRequest(HttpRequest):
    url = "https://example.com"
    show_request_headers = True
    show_response_headers = True
    show_response_body = True
```

## HTTP Server (for testing)

Kutsu includes a built-in Tornado-based HTTP server for mocking endpoints:

```python
from kutsu.http_server import HttpServer, HttpRequestHandler
import json

class TestServer(HttpServer):
    port = 9999

class HelloHandler(HttpRequestHandler):
    server = TestServer
    path = r"/hello"

    def get(self):
        self.write(json.dumps({"message": "hello"}))
        self.set_header("Content-Type", "application/json")

# Server auto-starts when the class is defined (autostart=True by default)
# Now you can make requests to http://localhost:9999/hello
```

## Chaining It All Together

Here's a more complete example combining state, expressions, actions, and HTTP requests:

```python
from kutsu import State, Var, Secret, HttpRequest, Action, If

class Login(HttpRequest):
    method = "POST"
    url = "https://api.example.com/login"
    json = {
        "username": Var("username"),
        "password": Var("password"),
    }
    quiet = True

    def on_response(self, state, response):
        state.token = response.json()["token"]
        return state

class GetProfile(HttpRequest):
    url = "https://api.example.com/profile"
    auth_token = Var("token")
    quiet = True

    def on_response(self, state, response):
        state.profile = response.json()
        return state

# Build a pipeline
pipeline = Login() >> GetProfile()

# Run it
result = State(
    username="alice",
    password=Secret("s3cret"),
) | pipeline

print(result.profile)
```
