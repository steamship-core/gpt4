import json

import pytest
from steamship import Block, Tag, MimeTypes, SteamshipError, File, Steamship
from steamship.data.tags.tag_constants import TagKind, RoleTag
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import (
    RawBlockAndTagPluginInput,
)
from steamship.plugin.inputs.raw_block_and_tag_plugin_input_with_preallocated_blocks import (
    RawBlockAndTagPluginInputWithPreallocatedBlocks,
)
from steamship.plugin.request import PluginRequest

from src.api import GPT4Plugin


@pytest.mark.parametrize("model", ["", "gpt-4-32k"])
def test_generator(model: str):
    gpt4 = GPT4Plugin(config={"n": 4, "model": model})

    blocks = [
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
    ]

    new_blocks = gpt4.run(PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks)))
    assert len(new_blocks.data.blocks) == 4
    for block in new_blocks.data.blocks:
        assert block.text.strip().startswith("5 6 7 8")

    assert new_blocks.data.usage is not None
    assert len(new_blocks.data.usage) == 2


def test_stopwords():
    gpt4 = GPT4Plugin(config={})

    blocks = [
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
    ]

    new_blocks = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInput(blocks=blocks, options={"stop": "6"})
        )
    )
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text.strip() == "5"


def test_functions():
    gpt4 = GPT4Plugin(
        config={},
    )

    blocks = [
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
    ]

    new_blocks = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInput(
                blocks=blocks,
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
        )
    )
    assert len(new_blocks.data.blocks) == 1
    assert "function_call" in new_blocks.data.blocks[0].text.strip()
    function_call = json.loads(new_blocks.data.blocks[0].text.strip())
    assert "function_call" in function_call


def test_functions_function_message():
    gpt4 = GPT4Plugin(
        config={},
    )

    blocks = [
        Block(
            text="You are a helpful AI assistant.",
            tags=[Tag(kind="role", name="system")],
        ),
        Block(
            text="Who is Vin Diesel's girlfriend?",
            tags=[Tag(kind="role", name="user")],
        ),
        Block(
            text='{"function_call": {"name": "Search", "arguments": "{\\n  \\"__arg1\\": \\"Vin Diesel\'s girlfriend\\"\\n}"}}',
            tags=[Tag(kind="role", name="assistant")],
        ),
        Block(
            text="Paloma Jiménez",
            tags=[Tag(kind="role", name="function"), Tag(kind="name", name="Search")],
        ),
    ]

    new_blocks = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInput(
                blocks=blocks,
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
        )
    )
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text is not None
    assert isinstance(new_blocks.data.blocks[0].text, str)
    text = new_blocks.data.blocks[0].text.strip()
    assert "Vin Diesel" in text


def test_default_prompt():
    gpt4 = GPT4Plugin(
        config={
            "openai_api_key": "",
            "default_system_prompt": "You are very silly and are afraid of numbers. When you see "
            "them you scream: 'YIKES!'",
            "moderate_output": False,
        }
    )

    blocks = [
        Block(
            text="1 2 3 4",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    new_blocks = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInput(blocks=blocks, options={"stop": "6"})
        )
    )
    assert len(new_blocks.data.blocks) == 1
    assert new_blocks.data.blocks[0].text.strip() == "YIKES!"


def test_flagged_prompt():
    gpt4 = GPT4Plugin(config={"openai_api_key": ""})

    blocks = [
        Block(
            text=" <Insert something super offensive here to run this test>",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    with pytest.raises(SteamshipError):
        new_blocks = gpt4.run(
            PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks))
        )


def test_invalid_model_for_billing():
    with pytest.raises(SteamshipError) as e:
        _ = GPT4Plugin(
            config={"model": "a model that does not exist", "openai_api_key": ""}
        )
        assert "This plugin cannot be used with model" in str(e)


def test_streaming_generation():
    client = Steamship(profile="test")
    gpt4 = GPT4Plugin(client=client, config={})

    blocks = [
        Block(
            text="Tell me a 500 word story about bananas",
            tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
            mime_type=MimeTypes.TXT,
        ),
    ]

    blocks_to_allocate = gpt4.determine_output_block_types(
        PluginRequest(data=RawBlockAndTagPluginInput(blocks=blocks, options={"n": 3}))
    )
    assert len(blocks_to_allocate.data.block_types_to_create) == 3
    file = File.create(client, blocks=[])
    output_blocks = []
    for block_type_to_allocate in blocks_to_allocate.data.block_types_to_create:
        assert block_type_to_allocate == MimeTypes.TXT.value
        output_blocks.append(
            Block.create(
                client,
                file_id=file.id,
                mime_type=MimeTypes.TXT.value,
                streaming=True,
            )
        )

    result_usage = gpt4.run(
        PluginRequest(
            data=RawBlockAndTagPluginInputWithPreallocatedBlocks(
                blocks=blocks, options={"n": 3}, output_blocks=output_blocks
            )
        )
    )
    result_texts = [fetch_result_text(block) for block in output_blocks]

    for i, text in enumerate(result_texts):
        print(f"{i} **********")
        print(text)
        print("\n\n\n\n")


def fetch_result_text(block: Block) -> str:
    bytes = block.raw()
    return str(bytes, encoding="utf-8")
