import pytest
from unittest.mock import MagicMock
import dspy
from dspy.signatures.signature import Signature
from src.human_annotate.human_annotation import Query, _form, FormData, Result
import threading
from typing import Literal

# Define a mock signature for testing
class MockSignature(Signature):
    """A mock signature for testing."""
    input_field: str = dspy.InputField(desc="This is an input field.")
    output_field: str = dspy.OutputField(desc="This is an output field.")

class MockSignatureWithEnum(Signature):
    """A mock signature for testing with enums."""
    input_field: str = dspy.InputField(desc="This is an input field.")
    output_field: Literal["A", "B", "C"] = dspy.OutputField(desc="This is an output field.")

# Test cases

def test_form_respects_signature():
    """
    - The form correctly respects the signature.
    - All inputs are present on the page.
    - All outputs are present in the form on the page.
    """
    inputs = FormData(input_field="test_input")
    filled_outputs = FormData()
    errors = {}
    
    form_component = _form(MockSignature, inputs, filled_outputs, errors)
    form_html = str(form_component)
    
    assert "input_field" in form_html
    assert "test_input" in form_html
    assert "output_field" in form_html

def test_form_fails_with_missing_inputs():
    """
    - If not all inputs are provided, it should fail.
    """
    with pytest.raises(ValueError, match="Missing input field 'input_field' in data"):
        Query(MockSignature, {})

def test_form_displays_errors():
    """
    - It correctly displays errors.
    """
    inputs = FormData(input_field="test_input")
    filled_outputs = FormData(output_field="some_value")
    errors = {"output_field": "Invalid value"}
    
    form_component = _form(MockSignature, inputs, filled_outputs, errors)
    form_html = str(form_component)
    
    assert "Invalid value" in form_html
    assert "text-red-500" in form_html

def test_form_displays_filled_fields():
    """
    - It correctly displays already filled fields.
    """
    inputs = FormData(input_field="test_input")
    filled_outputs = FormData(output_field="filled_value")
    errors = {}
    
    form_component = _form(MockSignature, inputs, filled_outputs, errors)
    form_html = str(form_component)

    assert "filled_value" in form_html

def test_query_returns_form():
    """
    - It returns a form with inputs and outputs.
    """
    query = Query(MockSignature, {"input_field": "test_input"})
    form_component = query.get()
    form_html = str(form_component)
    
    assert "input_field" in form_html
    assert "test_input" in form_html
    assert "output_field" in form_html

def test_query_validation_failure():
    """
    - If validation fails, it should stay on the same page.
    """
    query = Query(MockSignature, {"input_field": "test_input"})
    form_data = FormData(output_field='{"test": "test"}') # Invalid json
    
    # Mock the event
    query_processed = threading.Event()
    
    next_state, _ = query.post(query_processed, form_data)
    
    assert next_state == query
    assert not query_processed.is_set()

def test_query_validation_success():
    """
    - If successful, it returns a result state with the correct completions.
    """
    query = Query(MockSignature, {"input_field": "test_input"})
    form_data = FormData(output_field='"valid_output"')
    
    # Mock the event
    query_processed = threading.Event()
    
    next_state, _ = query.post(query_processed, form_data)
    
    assert isinstance(next_state, Result)
    assert next_state.prediction.output_field == "valid_output"
    assert query_processed.is_set()

def test_form_with_enum():
    """
    - It correctly displays radio buttons for enum fields.
    """
    inputs = FormData(input_field="test_input")
    filled_outputs = FormData(output_field='"B"')
    errors = {}
    
    form_component = _form(MockSignatureWithEnum, inputs, filled_outputs, errors)
    form_html = str(form_component)

    assert 'type="radio"' in form_html
    assert 'value=\'"A"\'' in form_html
    assert 'value=\'"B"\'' in form_html
    assert 'value=\'"C"\'' in form_html
    assert 'checked' in form_html
    assert 'value=\'"B"\' checked' in form_html
