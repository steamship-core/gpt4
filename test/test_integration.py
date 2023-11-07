"""Test gpt-4 generation via integration tests."""
import json
from typing import Optional

import pytest
from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag, TagValueKey

GENERATOR_HANDLE = "gpt-4"

@pytest.mark.parametrize(
    "model", ["", "gpt-4-32k", "gpt-4-1106-preview"]
)
def test_generator(model: str):
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE, config={"model": model})
        file = File.create(
            steamship,
            blocks=[
                Block(
                    text="You are an assistant who loves to count",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                    mime_type=MimeTypes.TXT,
                ),
                Block(
                    text="Continue this series: 1 2 3 4",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                    mime_type=MimeTypes.TXT,
                ),
            ],
        )

        generate_task = gpt4.generate(input_file_id=file.id)
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        for block in output.blocks:
            assert block.text.strip().startswith("5")
            assert block.text.strip() == "5 6 7 8"


def test_generator_without_role():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        file = File.create(
            steamship,
            blocks=[
                Block(text="1 2 3 4"),
            ],
        )
        generate_task = gpt4.generate(input_file_id=file.id)
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1


def test_stopwords():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        file = File.create(
            steamship,
            blocks=[
                Block(
                    text="You are an assistant who loves to count",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                    mime_type=MimeTypes.TXT,
                ),
                Block(
                    text="Continue this series: 1 2 3 4",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                    mime_type=MimeTypes.TXT,
                ),
            ],
        )
        generate_task = gpt4.generate(
            input_file_id=file.id, options={"stop": ["6", "7"]}
        )
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        for block in output.blocks:
            assert block.text.strip() == "5"


def test_functions():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        file = File.create(
            steamship,
            blocks=[
                Block(
                    text="You are a helpful AI assistant.",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                    mime_type=MimeTypes.TXT,
                ),
                Block(
                    text="Search for the weather of today in Berlin",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                    mime_type=MimeTypes.TXT,
                ),
            ],
        )
        generate_task = gpt4.generate(
            input_file_id=file.id,
            options={
                "functions": [
                    {
                        "name": "Search",
                        "description": "useful for when you need to answer questions about current events. You should ask targeted questions",
                        "parameters": {
                            "properties": {
                                "query": {"title": "query", "type": "string"}
                            },
                            "required": ["query"],
                            "type": "object",
                        },
                    }
                ]
            },
        )
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        assert "function_call" in output.blocks[0].text.strip()
        function_call = json.loads(output.blocks[0].text.strip())
        assert "function_call" in function_call


def test_multimodal_functions_with_blocks():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        file = File.create(
            steamship,
            blocks=[
                Block(
                    text="You are a helpful AI assistant.",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                    mime_type=MimeTypes.TXT,
                ),
                Block(
                    text="Generate an image of a sailboat.",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                    mime_type=MimeTypes.TXT,
                ),
                Block(
                    text=json.dumps({"name": "generate_image", "arguments": "{ \"text\": \"sailboat\" }"}),
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.ASSISTANT),
                          Tag(kind="function-selection", name="generate_image")],
                    mime_type=MimeTypes.PNG,
                ),
                Block(
                    text="c2f6818c-233d-4426-9dc5-f3c28fa33068",
                    tags=[Tag(kind=TagKind.ROLE, name="function", value={TagValueKey.STRING_VALUE: "generate_image"})],
                    mime_type=MimeTypes.PNG,
                ),
                Block(
                    text="Make the background of the image blue.",
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                    mime_type=MimeTypes.TXT,
                ),
            ],
        )
        generate_task = gpt4.generate(
            input_file_id=file.id,
            options={
                "functions": [
                    {
                        "name": "PixToPixTool",
                        "description": "Modifies an existing image according to a text prompt.",
                        "parameters": {
                         "type": "object",
                         "properties": {
                             "text": {
                                 "type": "string",
                                 "description": "text prompt for a tool.",
                             },
                             "uuid": {
                                 "type": "string",
                                 "description": "UUID for a Steamship Block. Used to refer to a non-textual input generated by another function. Example: \"c2f6818c-233d-4426-9dc5-f3c28fa33068\""
                             }
                         }
                        }
                     },
                    {
                        "name": "DalleTool",
                        "description": "Generates a new image from a text prompt.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "text prompt for a tool.",
                                },
                                "uuid": {
                                    "type": "string",
                                    "description": "UUID for a Steamship Block. Used to refer to a non-textual input generated by another function. Example: \"c2f6818c-233d-4426-9dc5-f3c28fa33068\""
                                }
                            }
                        }
                    }
                ]
            },
        )
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        assert "function_call" in output.blocks[0].text.strip()
        function_call = json.loads(output.blocks[0].text.strip())
        assert "function_call" in function_call
        fc = function_call.get('function_call')
        assert "PixToPixTool" == fc.get('name', "")
        args = fc.get("arguments", None)
        assert args is not None
        assert 'uuid' in args
        assert "c2f6818c-233d-4426-9dc5-f3c28fa33068" in args
        assert 'text' in args
        assert "blue" in args


def test_functions_function_message():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)

        file = File.create(
            steamship,
            blocks=[
                Block(
                    text="You are a helpful AI assistant.",
                    tags=[Tag(kind="role", name="system")],
                ),
                Block(
                    text="Who is Vin Diesel's girlfriend?",
                    tags=[Tag(kind="role", name="user")],
                ),
                Block(
                    text=json.dumps({"name": "Search", "arguments": "{ \"query\": \"Vin Diesel\'s girlfriend\" }"}),
                    tags=[Tag(kind=TagKind.ROLE, name=RoleTag.ASSISTANT),
                          Tag(kind="function-selection", name="Search")],
                ),
                Block(
                    text="Paloma Jiménez",
                    tags=[
                        Tag(kind="role", name="function"),  # legacy method left for testing / backwards compat
                        Tag(kind="name", name="Search"),
                    ],
                ),
            ],
        )

        generate_task = gpt4.generate(
            input_file_id=file.id,
            options={
                "functions": [
                    {
                        "name": "Search",
                        "description": "useful for when you need to answer questions about current events. You should ask targeted questions",
                        "parameters": {
                            "properties": {
                                "query": {"title": "query", "type": "string"}
                            },
                            "required": ["query"],
                            "type": "object",
                        },
                    }
                ]
            },
        )
        generate_task.wait()
        output = generate_task.output
        assert len(output.blocks) == 1
        assert output.blocks[0].text is not None
        assert isinstance(output.blocks[0].text, str)
        text = output.blocks[0].text.strip()
        assert "Vin Diesel" in text
