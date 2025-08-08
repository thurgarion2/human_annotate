from fasthtml import FastHTML
from fasthtml.common import *
from fasthtml.components import *
import uvicorn
from queue import Queue, Empty
import threading
from dspy.signatures.signature import Signature
import dspy
from typing import Tuple, Any, Mapping
from pydantic import TypeAdapter, ValidationError, BaseModel
import json

def _form(signature:type[Signature], inputs_data:Mapping, filled_outputs:dict[str,str]={}, outputs_errors:dict[str,str]={}):
    """return a FT Components of the form"""
    input_displays = [H2("Inputs", cls="text-2xl font-bold mb-4")]
    
    for name, value in inputs_data.items():
        field_desc = signature.input_fields[name].description or name
        input_displays.extend([
            H3(field_desc, cls="text-xl font-semibold mt-4"),
            P(str(value), cls="text-gray-700")
        ])

    form_fields = []
    for name, field in signature.output_fields.items():
        field_id = name
        error = outputs_errors.get(name)
        value = filled_outputs.get(name, "")
        schema = TypeAdapter(field.annotation).json_schema()

        if schema.get('enum'):
            form_fields.append(Label(schema.get('title', name), fr=field_id, cls="block text-gray-700 text-sm font-bold mb-2"))
            for v in schema['enum']:
                is_checked = value == str(v)
                form_fields.extend([
                    Input(type="radio", id=f"{field_id}_{v}", name=field_id, value=v, checked=is_checked, cls="mr-2"),
                    Label(str(v), fr=f"{field_id}_{v}", cls="mr-4")
                ])
        else:
            form_fields.extend([
                Label(f"{name}: {schema.get('properties','')}", fr=field_id, cls="block text-gray-700 text-sm font-bold mb-2"),
                Textarea(value, id=field_id, name=field_id, cls="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline")
            ])

        if error:
            form_fields.append(P(error, cls="text-red-500 text-xs italic mt-1"))
        form_fields.append(Br())

    return Div(
        H1("Annotation Request", cls="text-4xl font-bold mb-8"),
        Div(*input_displays, cls="mb-8"),
        P("Please provide the following information:", cls="text-gray-700 mb-4"),
        Form(
            *form_fields,
            Input(type="submit", value="Submit", cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"),
            hx_post="/",
            hx_target="body"
        ),
        cls="max-w-2xl mx-auto p-8 bg-white rounded-lg shadow-lg"
    )

class HumanPredict(dspy.Module):
    """query a human annotator to fill the prediction instead of an llm, used to annotate example"""

    def __init__(self, signature: type[Signature], server):
        super().__init__()
        self.signature = signature
        self.server = server

    def forward(self, *args, **kwargs) -> dspy.Prediction:
        prediction_dict = self.server.query(self.signature, **kwargs)
        return dspy.Prediction(**prediction_dict)

class ChatServer:
    """a fasthtml chat application for now it display a page that is either in waiting state if there is no request or display a request to fill"""

    def __init__(self, port: int):
        self.port = port
        self.app = FastHTML(hdrs=[
            Script(src="https://cdn.tailwindcss.com"),
            Title("Human Annotation")
        ])
        self.server_thread = None
        self.is_running = False
        self.state : Any = Waiting()
        self.lock = threading.Lock()
        self.query_processed = threading.Event()

        @self.app.get("/")
        def get(request):
            return Body(self.state.get(), cls="bg-gray-100")

        @self.app.post("/")
        async def post(request):
            data = {}
            async with request.form() as form:
                data = dict(form.items())
                
            with self.lock:
                next_state, html = self.state.post(self.query_processed, data)
                self.state = next_state
            return Body(html, cls="bg-gray-100")

    def query(self, signature: type[Signature], **data) -> dict[str, Any]:
        """query the human annotar, validate the response and return the result"""
        with self.lock:
            self.state = Query(signature, data)
            self.query_processed.clear()
            
        self.query_processed.wait()
        with self.lock:
            result = self.state.prediction
            self.state = Waiting()
            self.query_processed.clear()
            
        return result

    def start(self):
        if not self.is_running:
            self.is_running = True
            config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port)
            server = uvicorn.Server(config)
            self.server_thread = threading.Thread(target=server.run)
            self.server_thread.start()
            print(f"Server started on port {self.port}")

    def stop(self):
        if self.is_running:
            self.is_running = False
            print("Server stopping...")


class Query:
    
    def __init__(self, signature: type[Signature], data: Mapping, errors: dict | None = None, form_data: dict | None = None) -> None:
        for name in signature.input_fields.keys():
            if name not in data:
                raise ValueError(f"Missing input field '{name}' in data")
        self.signature = signature
        self.data = data
        self.errors = errors or {}
        self.form_data = form_data or {}

    def get(self):
        return _form(self.signature, self.data, self.form_data, self.errors)

    def post(self, query_processed: threading.Event, formData: Mapping) -> Tuple[Any, Any]:
        parsed_data = {}
        errors = {}

        for key, field in self.signature.output_fields.items():
            adapter = TypeAdapter(field.annotation)
            value = formData.get(key)
            try:
                if issubclass(field.annotation, BaseModel):
                    parsed_data[key] = adapter.validate_json(value)
                else:
                    parsed_data[key] = adapter.validate_python(value)
            except ValidationError as e:
                errors[key] = e.errors()[0]["msg"]
            except Exception as e:
                errors[key] = str(e)

        if errors:
            next_state = Query(self.signature, self.data, errors=errors, form_data=formData)
            return next_state, next_state.get()

        result = Result(dspy.Prediction.from_completions({key:[value] for key, value in parsed_data.items()}, signature=self.signature))
        query_processed.set()
        return result, result.get()

class Result:
    def __init__(self, prediction: dspy.Prediction):
        self.prediction = prediction
        
    def get(self):
        return Div(P("Annotation submitted. Waiting for next request...", cls="text-lg text-gray-700"), cls="max-w-2xl mx-auto p-8 bg-white rounded-lg shadow-lg text-center")
    
    def post(self, query_processed: threading.Event, **kwargs) -> Tuple[Any, Any]:
        return self, self.get()

class Waiting:
    
    def get(self):
        return Div(P("Annotation submitted. Waiting for next request...", cls="text-lg text-gray-700"), cls="max-w-2xl mx-auto p-8 bg-white rounded-lg shadow-lg text-center")
    
    def post(self, query_processed: threading.Event, **kwargs) -> Tuple[Any, Any]:
        return self, self.get()
