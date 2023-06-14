"""Test gpt-4 generation via integration tests."""
import json

from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag

GENERATOR_HANDLE = "gpt-4"


def test_generator():
    with Steamship.temporary_workspace() as steamship:
        gpt4 = steamship.use_plugin(GENERATOR_HANDLE)
        print(gpt4)
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
            assert block.text.strip() == "5 6 7 8 9 10"


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
                    text="Search for the weather of today",
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
